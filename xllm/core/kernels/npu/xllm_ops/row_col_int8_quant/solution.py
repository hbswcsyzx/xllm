# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Public ACLNN and direct-kernel entry points; build before importing."""

import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch_npu

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("rowcol_native", _ROOT / "rowcol_native.so")
_NATIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NATIVE)


def _validate(inputs: list[torch.Tensor], attributes: dict[str, Any]) -> torch.Tensor:
    if not isinstance(attributes, dict) or attributes:
        raise ValueError("attributes must be an empty dictionary")
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
        raise ValueError("expected exactly one input")
    x = inputs[0]
    if not isinstance(x, torch.Tensor) or x.device.type != "npu":
        raise ValueError("x must be an NPU tensor")
    if x.dtype != torch.float16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be contiguous ND float16[M,N]")
    m, n = x.shape
    if not 1 <= m <= 4096 or not 2 <= n <= 4096 or n % 2:
        raise ValueError("M in [1,4096], N even in [2,4096] required")
    return x


def prepare(inputs: list[torch.Tensor], attributes: dict[str, Any]) -> None:
    """Validate metadata and warm runtime caches without retaining outputs."""
    x = _validate(inputs, attributes)
    _NATIVE.run(x)


def run(inputs: list[torch.Tensor], attributes: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return _NATIVE.run_checked(inputs, attributes)


def run_direct(inputs: list[torch.Tensor], attributes: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return _NATIVE.run_direct(_validate(inputs, attributes), str(_ROOT / "artifacts/kernel.o"))
