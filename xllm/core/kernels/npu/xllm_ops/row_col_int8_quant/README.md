<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->
# RowColInt8Quant

ACLNN implementation for the xLLM October 2026 community task. Input is a
finite contiguous ND FP16 matrix `[M,N]`, `1<=M<=4096`, even `2<=N<=4096`.
Returns independent contiguous NPU tensors `qRow:int8[M,N]`,
`qCol:int8[M,N]`, `rowAbsmax:float[M]`, `colAbsmax:float[N]`.
Statistics are absolute maxima. Quantization uses FP32 `127/s`, FP32 multiply,
ties-to-even rounding and clipping to [-127,127]; zero axes produce zero.

The device code uses compiler intrinsics directly. M<512 uses grouped rows
and independent 128x64 column chunks. M>=512 uses 64x256 partial maxima,
two hardware publication barriers, and tiled quantization. Host tiling selects
one compiled key; each generated kernel has one compute schedule. No value-
dependent branch or case-name routing. See [design](docs/design.md) and
[API evidence](docs/api-evidence.md).

Tested with CANN 9.1.0 on Ascend910_9362, driver 25.5.5, Python 3.12.13,
torch 2.10.0+cpu and torch_npu 2.10.0.post4+git5dd8ef3. The task specifies
9382/48 AIV, while this allocated device exposes 40 AIV. Results retain the
actual hardware identity; external acceptance on the specified hardware is
separate from these local tests.

From the task worktree, always enter its official-image environment:

```bash
op=xllm/core/kernels/npu/xllm_ops/row_col_int8_quant
../run.sh bash "$op/build.sh"
../run.sh python3 "$op/build_bridge.py"
../run.sh bash "$op/examples/run.sh"
../run.sh python3 ../material/public/verify.py --solution-file "$op/solution.py" --report ../docs/public-final.json
../run.sh python3 "$op/tests/engineering.py" --public ../material/public --report ../docs/engineering-final.json
../run.sh python3 ../material/public/benchmark.py --solution-file "$op/solution.py" --report ../docs/performance-final.json
../run.sh python3 ../material/public/check_performance.py --report ../docs/performance-final.json --output ../docs/performance-acceptance.json
```

`build.sh` stages only this operator into a private copy of the repository's
pinned xllm_ops build framework, packages it and installs under
`/opt/asc-op-dev/vendors/rowcol_int8_quant_xllm_math`. It never installs into
system CANN. Build/cache paths are under the task's `state/build/` directory.
The source repository's submodule remains unchanged. CMake targets provide
equivalent package/bridge/example build entry points.

`prepare(inputs,{})` validates and warms caches without retaining results.
`run(inputs,{})` allocates fresh outputs and invokes `aclnnRowColInt8Quant`
through the framework's current-stream TaskQueue. Aligned calls reuse bounded
per-stream/shape ACLNN metadata via the documented repeatable-executor and
address-update APIs; each call updates addresses and launches computation.
Unaligned inputs use fresh phase-1 metadata. `run_direct(inputs,{})` serializes the
same tiling class and launches the installed compiler object using the ACL
host-argument kernel API. Both use the current NPU stream and protect input
lifetime with the NPU allocator. Workspace and outputs are allocated on the
consumer stream. Callbacks retain input/workspace through submission; output
reuse follows that same stream's allocator and TaskQueue order.
Direct-launch tiling upload
is not included in the formal performance comparison; official timing always
calls `run` with fresh outputs.

The frozen timing protocol is 50 warmups, 5 rounds of 200 calls, median of
round-mean API-to-completion wall times. Every wall/event round is retained.
Device Events include submission gaps. Profile task time is diagnostic only.
The fixed limits are 32/35/73/244/42 us for public cases 004..008. Both official
baseline candidates are independently accuracy-qualified and timed by
`tests/baselines.py` using the same protocol.

Local final results (2026-10-28): 27/27 public inputs and 12/12 extra offset
inputs passed; all 115 engineering records passed. Three consecutive complete
official performance runs passed. The first complete run is reported below;
rounds are not selected across runs.

| Case | Shape | Wall median (us) | Limit (us) |
| --- | --- | ---: | ---: |
| 004 | 257x1026 | 30.512105 | 32 |
| 005 | 128x4096 | 19.356120 | 35 |
| 006 | 1024x4096 | 63.445295 | 73 |
| 007 | 4096x4096 | 176.036895 | 244 |
| 008 | 4096x256 | 30.754765 | 42 |

The official seven-file reports, raw logs, private installation package and
prebuilt binaries are delivered separately as local acceptance materials.
They are excluded from this source branch. All maxima are bit-exact;
INT8 differences are restricted to the official half-integer neighborhood.
The design was submitted as [community PR 2513](https://gitcode.com/cann/cann-ops-competitions/merge_requests/2513)
by the registered GitCode account `daivernon`; review and platform acceptance
are pending. Independent coverage review corrected the prescribed rank input
to `(2,2,2)` and kept the noncontiguous input's shape `(257,1026)` while using
strides `(2052,2)`. All 115 engineering records passed again with input metadata
saved in the locally delivered negative-case logs. Computation sources and
tested binaries remain unchanged; their hashes and the fixed source commit
are recorded in the locally delivered acceptance manifest.

The task permits one-unit int8 differences only within its FP32 half-integer
neighborhood, with strict ties/zeros and bit-exact maxima. The unmodified
official verifier enforces that rule. Engineering tests cover the 12 exact
extra seed IDs with storage offset 1, 39 negative entry records, streams,
input/output lifetimes, active-memory reclamation and verifier counterexamples.
The standalone ACLNN example also rejects six invalid metadata configurations.

The workspace bound is `16 MiB + 4*ceil(M/64)*ceil(N/256)*320`.
No complete FP32 matrix or transposed matrix is allocated. Private installation
packages and complete result manifests are delivered separately from Git.
