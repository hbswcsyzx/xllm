// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>

#include "acl/acl.h"
#include "aclnn_row_col_int8_quant.h"
#include "op_host/plan.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
extern "C" size_t rowcol_make_tiling(uint32_t m,
                                     uint32_t n,
                                     uint32_t pad,
                                     uint64_t ffts,
                                     void* buffer,
                                     size_t capacity);

namespace {
thread_local uint64_t last_workspace_size = 0;
std::atomic<uint64_t> actual_workspace_size{0};
void check_acl(aclError error) {
  if (error != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error("ACL failure " + std::to_string(error) + ": " +
                             (detail ? detail : "no detail"));
  }
}
class Descriptor final {
 public:
  Descriptor(void* address, aclDataType dtype, c10::IntArrayRef sizes) {
    std::array<int64_t, 2> strides{sizes.size() == 2 ? sizes[1] : 1, 1};
    value_ = aclCreateTensor(sizes.data(),
                             sizes.size(),
                             dtype,
                             strides.data(),
                             0,
                             ACL_FORMAT_ND,
                             sizes.data(),
                             sizes.size(),
                             address);
    if (value_ == nullptr) {
      throw std::runtime_error("aclCreateTensor failed");
    }
  }
  Descriptor(const torch::Tensor& tensor, aclDataType dtype) {
    value_ = aclCreateTensor(tensor.sizes().data(),
                             tensor.dim(),
                             dtype,
                             tensor.strides().data(),
                             0,
                             ACL_FORMAT_ND,
                             tensor.sizes().data(),
                             tensor.dim(),
                             tensor.data_ptr());
    if (value_ == nullptr) {
      throw std::runtime_error("aclCreateTensor failed");
    }
  }
  ~Descriptor() { aclDestroyTensor(value_); }
  aclTensor* get() const { return value_; }

 private:
  aclTensor* value_;
};

class CachedPlan final {
 public:
  CachedPlan(const torch::Tensor& x, const std::array<void*, 4>& outputs)
      : dx_(x, ACL_FLOAT16),
        dqr_(outputs[0], ACL_INT8, x.sizes()),
        dqc_(outputs[1], ACL_INT8, x.sizes()),
        dr_(outputs[2], ACL_FLOAT, {x.size(0)}),
        dc_(outputs[3], ACL_FLOAT, {x.size(1)}) {
    check_acl(aclnnRowColInt8QuantGetWorkspaceSize(dx_.get(),
                                                   dqr_.get(),
                                                   dqc_.get(),
                                                   dr_.get(),
                                                   dc_.get(),
                                                   &workspace_size_,
                                                   &executor_));
    check_acl(aclSetAclOpExecutorRepeatable(executor_));
  }
  ~CachedPlan() { aclDestroyAclOpExecutor(executor_); }
  void update(const torch::Tensor& x, const std::array<void*, 4>& outputs) {
    check_acl(aclSetInputTensorAddr(executor_, 0, dx_.get(), x.data_ptr()));
    check_acl(aclSetOutputTensorAddr(executor_, 0, dqr_.get(), outputs[0]));
    check_acl(aclSetOutputTensorAddr(executor_, 1, dqc_.get(), outputs[1]));
    check_acl(aclSetOutputTensorAddr(executor_, 2, dr_.get(), outputs[2]));
    check_acl(aclSetOutputTensorAddr(executor_, 3, dc_.get(), outputs[3]));
  }
  uint64_t workspace_size() const { return workspace_size_; }
  aclOpExecutor* executor() const { return executor_; }

 private:
  Descriptor dx_, dqr_, dqc_, dr_, dc_;
  uint64_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
};

using PlanKey = std::tuple<int32_t, uintptr_t, int64_t, int64_t>;
thread_local std::map<PlanKey, std::unique_ptr<CachedPlan>> plans;
thread_local std::map<std::pair<int32_t, int64_t>, aclrtStream> streams;

int32_t launch_aclnn(const torch::Tensor& x,
                     const std::array<void*, 4>& outputs,
                     const torch::Tensor& workspace,
                     aclrtStream raw_stream) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  PlanKey key{x.device().index(),
              reinterpret_cast<uintptr_t>(raw_stream),
              x.size(0),
              x.size(1)};
  const bool can_reuse = reinterpret_cast<uintptr_t>(x.data_ptr()) % 32 == 0 &&
                         (plans.size() < 64 || plans.count(key));
  std::array<std::unique_ptr<Descriptor>, 5> temporary;
  if (can_reuse) {
    auto& plan = plans[key];
    if (!plan) {
      plan = std::make_unique<CachedPlan>(x, outputs);
    }
    plan->update(x, outputs);
    workspace_size = plan->workspace_size();
    executor = plan->executor();
  } else {
    temporary[0] = std::make_unique<Descriptor>(x, ACL_FLOAT16);
    temporary[1] =
        std::make_unique<Descriptor>(outputs[0], ACL_INT8, x.sizes());
    temporary[2] =
        std::make_unique<Descriptor>(outputs[1], ACL_INT8, x.sizes());
    temporary[3] = std::make_unique<Descriptor>(
        outputs[2], ACL_FLOAT, c10::IntArrayRef{x.size(0)});
    temporary[4] = std::make_unique<Descriptor>(
        outputs[3], ACL_FLOAT, c10::IntArrayRef{x.size(1)});
    check_acl(aclnnRowColInt8QuantGetWorkspaceSize(temporary[0]->get(),
                                                   temporary[1]->get(),
                                                   temporary[2]->get(),
                                                   temporary[3]->get(),
                                                   temporary[4]->get(),
                                                   &workspace_size,
                                                   &executor));
  }
  const uint64_t capacity = workspace.defined() ? workspace.nbytes() : 0;
  if (workspace_size != capacity) {
    throw std::runtime_error("ACLNN workspace size differs from host plan");
  }
  actual_workspace_size.store(workspace_size, std::memory_order_relaxed);
  check_acl(
      aclnnRowColInt8Quant(workspace.defined() ? workspace.data_ptr() : nullptr,
                           workspace_size,
                           executor,
                           raw_stream));
  return 0;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> run(
    const torch::Tensor& x) {
  if (x.device().type() != c10::DeviceType::PrivateUse1 ||
      x.scalar_type() != torch::kFloat16 || x.dim() != 2 ||
      !x.is_contiguous() || x.size(0) < 1 || x.size(0) > 4096 ||
      x.size(1) < 2 || x.size(1) > 4096 || x.size(1) % 2) {
    throw std::invalid_argument("invalid RowColInt8Quant input metadata");
  }
  auto qr = at_npu::native::OpPreparation::apply_tensor_without_format(
      x.sizes(), x.options().dtype(torch::kInt8));
  auto qc = at_npu::native::OpPreparation::apply_tensor_without_format(
      x.sizes(), x.options().dtype(torch::kInt8));
  auto r = at_npu::native::OpPreparation::apply_tensor_without_format(
      {x.size(0)}, x.options().dtype(torch::kFloat32));
  auto c = at_npu::native::OpPreparation::apply_tensor_without_format(
      {x.size(1)}, x.options().dtype(torch::kFloat32));
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto stream_key = std::make_pair(static_cast<int32_t>(x.device().index()),
                                   static_cast<int64_t>(stream.id()));
  auto found = streams.find(stream_key);
  if (found == streams.end()) {
    found = streams.emplace(stream_key, stream.stream()).first;
  }
  aclrtStream raw_stream = found->second;
  c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(), stream);
  last_workspace_size =
      row_col_int8_quant::workspace_bytes(x.size(0), x.size(1));
  torch::Tensor workspace;
  if (last_workspace_size) {
    workspace = at_npu::native::OpPreparation::apply_tensor_without_format(
        {static_cast<int64_t>(last_workspace_size)},
        x.options().dtype(torch::kUInt8));
  }
  // Outputs originate on this stream. Same-stream allocator reuse follows
  // TaskQueue order, as in the framework's standard ACLNN enqueue macro.
  std::array<void*, 4> output_addresses{
      qr.data_ptr(), qc.data_ptr(), r.data_ptr(), c.data_ptr()};
  std::function<int32_t()> launch =
      [x, output_addresses, workspace, raw_stream]() -> int32_t {
    return launch_aclnn(x, output_addresses, workspace, raw_stream);
  };
  static const std::string op_name = "aclnnRowColInt8Quant";
  at_npu::native::OpCommand::RunOpApiV3(
      op_name, launch, /*sync=*/false, /*task_stream=*/&stream);
  return {qr, qc, r, c};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
run_direct(const torch::Tensor& x, const std::string& binary_path) {
  if (x.device().type() != c10::DeviceType::PrivateUse1 ||
      x.scalar_type() != torch::kFloat16 || x.dim() != 2 ||
      !x.is_contiguous() || x.size(0) < 1 || x.size(0) > 4096 ||
      x.size(1) < 2 || x.size(1) > 4096 || x.size(1) % 2) {
    throw std::invalid_argument("invalid RowColInt8Quant input metadata");
  }
  const uint32_t m = x.size(0), n = x.size(1);
  const bool tiled = row_col_int8_quant::use_tiled(m, n);
  uint32_t pad = 64;
  while (pad < n) {
    pad *= 2;
  }
  void* hardware_sync = nullptr;
  check_acl(aclrtGetHardwareSyncAddr(&hardware_sync));
  auto tiling_cpu =
      torch::empty({64}, torch::TensorOptions().dtype(torch::kUInt8));
  rowcol_make_tiling(m,
                     n,
                     pad,
                     reinterpret_cast<uint64_t>(hardware_sync),
                     tiling_cpu.data_ptr(),
                     64);
  auto tiling_device = tiling_cpu.to(x.device());
  auto qr = torch::empty(x.sizes(), x.options().dtype(torch::kInt8));
  auto qc = torch::empty(x.sizes(), x.options().dtype(torch::kInt8));
  auto r = torch::empty({m}, x.options().dtype(torch::kFloat32));
  auto c = torch::empty({n}, x.options().dtype(torch::kFloat32));
  uint64_t workspace_size = row_col_int8_quant::workspace_bytes(m, n);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                x.options().dtype(torch::kUInt8));
  static std::map<int32_t, aclrtBinHandle> binaries;
  aclrtBinHandle& binary = binaries[x.device().index()];
  if (binary == nullptr) {
    check_acl(aclrtBinaryLoadFromFile(binary_path.c_str(), nullptr, &binary));
  }
  aclrtFuncHandle function = nullptr;
  check_acl(aclrtBinaryGetFunctionByEntry(binary, tiled ? 1 : 0, &function));
  int64_t physical_cores = 0;
  check_acl(aclrtGetDeviceInfo(
      x.device().index(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &physical_cores));
  uint32_t cores =
      std::min<uint32_t>(physical_cores, std::max<uint32_t>(m, (n + 63) / 64));
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  for (const auto& tensor : {x, qr, qc, r, c, workspace, tiling_device}) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(),
                                               stream);
  }
  std::array<void*, 8> args = {hardware_sync,
                               x.data_ptr(),
                               qr.data_ptr(),
                               qc.data_ptr(),
                               r.data_ptr(),
                               c.data_ptr(),
                               workspace.data_ptr(),
                               tiling_device.data_ptr()};
  aclrtLaunchKernelAttr attribute{};
  attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attribute.value.schemMode = 1;
  aclrtLaunchKernelCfg config{&attribute, 1};
  check_acl(aclrtLaunchKernelWithHostArgs(function,
                                          cores,
                                          stream.stream(),
                                          tiled ? &config : nullptr,
                                          args.data(),
                                          sizeof(args),
                                          nullptr,
                                          0));
  last_workspace_size = workspace_size;
  actual_workspace_size.store(workspace_size, std::memory_order_relaxed);
  return {qr, qc, r, c};
}
}  // namespace
PYBIND11_MODULE(rowcol_native, module) {
  module.def("run", &run, pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "run_checked",
      [](const pybind11::object& inputs, const pybind11::object& attributes) {
        if (!pybind11::isinstance<pybind11::dict>(attributes) ||
            pybind11::len(attributes) != 0) {
          throw std::invalid_argument("attributes must be an empty dictionary");
        }
        if ((!pybind11::isinstance<pybind11::list>(inputs) &&
             !pybind11::isinstance<pybind11::tuple>(inputs)) ||
            pybind11::len(inputs) != 1) {
          throw std::invalid_argument("expected exactly one input");
        }
        auto sequence = inputs.cast<pybind11::sequence>();
        auto x = sequence[0].cast<torch::Tensor>();
        pybind11::gil_scoped_release release;
        return run(x);
      });
  module.def("run_direct",
             &run_direct,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("workspace_size", []() {
    return actual_workspace_size.load(std::memory_order_relaxed);
  });
}
