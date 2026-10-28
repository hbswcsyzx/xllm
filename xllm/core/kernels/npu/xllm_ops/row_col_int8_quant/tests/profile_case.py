# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Representative-case profiling only; not acceptance timing."""

import argparse
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()
    torch.npu.set_device(0)
    x = torch.randn((args.m, args.n), dtype=torch.float16, device="npu:0")
    for _ in range(3):
        solution.run([x], {})
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
