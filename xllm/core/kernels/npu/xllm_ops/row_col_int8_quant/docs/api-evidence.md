<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->
# CANN 9.1.0 API and synchronization evidence

Paths below are relative to `/usr/local/Ascend/cann-9.1.0` in the official
container, not the outer CANN installation. They describe 2201 only.

* [header] `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h`
  declares the compiler builtins. Raw UB/GM addresses are non-owning address
  space pointers, all UB DMA addresses 32-byte aligned.
* [impl] `aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h`:
  `_align_b16/b32/b8` burst length and GM gaps are bytes; UB gaps are 32-byte
  blocks. Normal copy burst length/gaps are 32-byte blocks. Tail input rows
  are zero initialized and DMA rounds each UB row to the next block.
* [impl] `python/site-packages/tbe/tikcpp/compile_op.py` generates
  `usrWorkspace=workspace+RESERVED_WORKSPACE` and
  `ascendc_compile_gen_code.py:gen_call_user_kernel` passes that pointer to the
  source kernel. The kernel must not add another 16 MiB. Host allocation
  includes the reserved prefix exactly once.
* [generated/impl] key 1 is `KERNEL_TYPE_MIX_AIV_1_0`, ratio 0:1 and
  crossCoreSync=1. `compile_op.py:_gen_kernel_func_declare_head` adds the
  hardware synchronization address before tensor pointers in the binary ABI.
  The direct entry supplies that leading pointer as well as five tensor
  pointers, workspace and tiling. `aclrtLaunchKernelWithHostArgs` copies the
  host argument buffer; `aclrtLaunchKernelV2` is not interchangeable.
* [header/impl] `include/acl/acl_rt.h:aclrtGetHardwareSyncAddr`, host schedule
  mode 1 and the runtime's physical AIV count are used for concurrent resident
  participants. All participants execute both flags. The raw sequence from
  `dav_c220/kernel_operator_sync_impl.h` is `ffts_cross_core_sync(PIPE_MTE3,
  1+(flag<<8)); wait_flag_dev(flag)`, flags 14 and 15, mode 0.
* [impl] `.../npu_arch_2201/vector_compute_impl/vconv/asc_half2int8_impl.h`
  uses destination/source repeat strides **4/8** for counter-mode FP16->INT8.
  A normal-mask 64-lane sequence instead uses 2/4. Reusing the latter strides
  in counter mode overlaps output repeats. FP32->INT32 RINT precedes narrowing;
  subsequent half values are exact small integers.
* [impl] `.../vector_compute_impl/asc_brcb_impl.h` and
  `dav_c220/kernel_operator_vec_reduce_impl.h`: BRCB takes unsigned bit views,
  dst block/repeat strides 1/8, eight repeats for 64 row scalars; vcmax
  ONLY_VALUE emits contiguous scalar maxima with destination stride 1.
* [impl] `.../vector_compute_impl/asc_div_impl.h` maps FP32 division to vdiv.
  Exact maxima and accepted half-neighborhood quantization differences are
  checked by the official independent CPU oracle; no global +/-1 tolerance.

Synchronization ownership: an input UB bank is free for MTE2 after its final
Vector reader; an output or partial-statistic bank is free after MTE3. Local
events represent MTE2->V, V->MTE2, V->MTE3, MTE3->V. The small kernel has an
explicit V->S event for its scale read. PIPE_V fences separate dependent
Vector instructions. The tiled kernel publishes only after its MTE3 writes,
then consumes published data on MTE2 after the corresponding all-core wait.

Host/framework evidence:

* [header/docs] `include/aclnn/acl_meta.h` declares
  `aclSetAclOpExecutorRepeatable`, `aclSetInputTensorAddr` and
  `aclSetOutputTensorAddr`. The executor is marked repeatable after its first
  workspace query; every subsequent launch updates the input and all output
  pointers. The address-update contract preserves alignment, so unaligned
  offset inputs use new descriptors and a fresh executor instead.
* [impl] The installed torch_npu `OpCommand::RunOpApiV3` queues the ACLNN
  callback on the framework TaskQueue. `NPUStream::stream()` drains pending
  tasks; caching the raw handle by current framework stream avoids draining
  on every call, while callback order remains under the TaskQueue. V3 accepts
  the current framework stream explicitly. The callable is stored directly
  as `std::function`, avoiding a second lambda-to-function copy; the operation
  name has process lifetime. Tensor captures own input and workspace arguments.
  Source: [pinned OpCommand.cpp](https://github.com/Ascend/pytorch/blob/5dd8ef3/torch_npu/csrc/framework/OpCommand.cpp).
* [impl/measured] The allocator's `recordStream` protects an input that may be
  released immediately on another stream. Workspace and outputs originate
  on the consumer stream. The callback holds input/workspace and captures
  output addresses by value, matching `EXEC_NPU_CMD` in the installed
  `include/third_party/op-plugin/op_plugin/utils/op_api_common.h`.
  Same-stream allocator order protects output reuse after its queued writer.
  Stream/lifetime/memory tests
  check these assumptions, including release pressure from a second stream.
* [measured] Final small-height profiling measured hot 257x1026 kernel tasks
  at 18.28--18.52 us after row/DMA batching, compared with 23.38--23.78 us
  before it. MTE3 time and vector time decreased. Complete-call wall timing
  remains the independent official performance metric.

[generated limitation] Installed llvm-objdump currently prints `<not available>`
for these ELF instruction words, including its advertised dav-c220-vec target.
msobjdump confirms entries and kernel types. No claim about exact emitted
instruction counts is based on the unsuccessful disassembly. Successful
ordinary msprof PMU data and same-device complete-call measurements are used
for performance conclusions. Failed msprof-op collection and compiler
save-temporary attempts are retained in the task evidence logs.
