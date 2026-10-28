# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Compile the thin bridge in the official container."""

import os
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
VENDOR = Path("/opt/asc-op-dev/vendors/rowcol_int8_quant_xllm_math")
NPU = Path(torch_npu.__file__).parent
CANN = Path(os.environ["ASCEND_HOME_PATH"])
os.environ["MAX_JOBS"] = "2"
module = load(
    name="rowcol_native",
    sources=[str(ROOT / "bridge.cpp"), str(ROOT / "tiling_bridge.cpp")],
    extra_include_paths=[
        str(ROOT),
        str(NPU / "include"),
        str(CANN / "include"),
        str(VENDOR / "op_api/include/aclnnop"),
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_ldflags=[
        f"-L{NPU / 'lib'}",
        "-ltorch_npu",
        f"-L{CANN / 'lib64'}",
        "-lascendcl",
        "-lregister",
        f"-L{VENDOR / 'op_api/lib'}",
        "-lcust_opapi",
        f"-Wl,-rpath,{VENDOR / 'op_api/lib'}",
    ],
    verbose=True,
)
(ROOT / "rowcol_native.so").write_bytes(Path(module.__file__).read_bytes())
