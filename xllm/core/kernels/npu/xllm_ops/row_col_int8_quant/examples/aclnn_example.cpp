// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "aclnn_row_col_int8_quant.h"

namespace {
void check(aclError error) {
  if (error != ACL_SUCCESS) {
    std::fprintf(stderr, "ACL error %d: %s\n", error, aclGetRecentErrMsg());
    std::exit(1);
  }
}
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(size_t bytes) {
    check(aclrtMalloc(&pointer_, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  ~DeviceBuffer() { aclrtFree(pointer_); }
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};
using Tensor = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
Tensor matrix(void* pointer, int64_t m, int64_t n, aclDataType dtype) {
  const int64_t shape[] = {m, n}, strides[] = {n, 1};
  return Tensor(
      aclCreateTensor(
          shape, 2, dtype, strides, 0, ACL_FORMAT_ND, shape, 2, pointer),
      aclDestroyTensor);
}
Tensor vector(void* pointer, int64_t size) {
  const int64_t shape[] = {size}, strides[] = {1};
  return Tensor(
      aclCreateTensor(
          shape, 1, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, 1, pointer),
      aclDestroyTensor);
}
}  // namespace

int main() {
  check(aclInit(nullptr));
  check(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  check(aclrtCreateStream(&stream));
  {
    const std::array<uint16_t, 8> host = {
        0x57f0, 0x3800, 0x3e00, 0x4100, 0xb800, 0xbe00, 0xc100, 0};
    DeviceBuffer x(16), qr(8), qc(8), r(4), c(32);
    check(aclrtMemcpy(x.get(), 16, host.data(), 16, ACL_MEMCPY_HOST_TO_DEVICE));
    auto dx = matrix(x.get(), 1, 8, ACL_FLOAT16);
    auto dqr = matrix(qr.get(), 1, 8, ACL_INT8),
         dqc = matrix(qc.get(), 1, 8, ACL_INT8);
    auto dr = vector(r.get(), 1), dc = vector(c.get(), 8);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    check(aclnnRowColInt8QuantGetWorkspaceSize(dx.get(),
                                               dqr.get(),
                                               dqc.get(),
                                               dr.get(),
                                               dc.get(),
                                               &workspace_size,
                                               &executor));
    DeviceBuffer workspace(workspace_size + 32);
    check(aclnnRowColInt8Quant(
        workspace.get(), workspace_size, executor, stream));
    check(aclrtSynchronizeStream(stream));
    std::array<int8_t, 8> actual_row{}, actual_col{};
    check(aclrtMemcpy(
        actual_row.data(), 8, qr.get(), 8, ACL_MEMCPY_DEVICE_TO_HOST));
    check(aclrtMemcpy(
        actual_col.data(), 8, qc.get(), 8, ACL_MEMCPY_DEVICE_TO_HOST));
    const std::array<int8_t, 8> expected_row = {127, 0, 2, 2, 0, -2, -2, 0};
    const std::array<int8_t, 8> expected_col = {
        127, 127, 127, 127, -127, -127, -127, 0};
    if (actual_row != expected_row || actual_col != expected_col) {
      return 2;
    }
    uint32_t rejected = 0;
    const std::array<std::array<int64_t, 2>, 6> invalid = {
        {{0, 8}, {1, 0}, {4097, 8}, {1, 4098}, {1, 3}, {1, 8}}};
    for (size_t i = 0; i < invalid.size(); ++i) {
      auto bad = matrix(x.get(),
                        invalid[i][0],
                        invalid[i][1],
                        i == 5 ? ACL_FLOAT : ACL_FLOAT16);
      aclOpExecutor* bad_executor = nullptr;
      aclError result = aclnnRowColInt8QuantGetWorkspaceSize(bad.get(),
                                                             dqr.get(),
                                                             dqc.get(),
                                                             dr.get(),
                                                             dc.get(),
                                                             &workspace_size,
                                                             &bad_executor);
      if (result == ACL_SUCCESS) {
        std::fprintf(stderr, "Invalid metadata %zu accepted\n", i);
        return 3;
      }
      ++rejected;
    }
    std::printf(
        "{\"status\":\"passed\",\"entry\":\"standalone_aclnn\",\"ties\":true,"
        "\"metadata_rejections\":%u}\n",
        rejected);
  }
  check(aclrtDestroyStream(stream));
  check(aclrtResetDevice(0));
  check(aclFinalize());
  return 0;
}
