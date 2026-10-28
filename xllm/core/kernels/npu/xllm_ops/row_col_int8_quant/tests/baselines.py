# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Measure both task-defined baselines with the frozen public protocol."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch_npu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.public.resolve()))
    import benchmark
    import generate_data
    import golden
    import verify

    torch.npu.set_device(0)
    records = []
    for case in json.loads((args.public / "case.json").read_text()):
        if not case["performance"]:
            continue
        arrays = generate_data.generate(case)
        inputs = verify.device_inputs(case, arrays, "npu:0")
        x = inputs[0]

        def composition(x: torch.Tensor = x) -> tuple:
            a = x.float()
            absolute = a.abs()
            r = absolute.amax(dim=1)
            c = absolute.amax(dim=0)

            def quant(scale: torch.Tensor) -> torch.Tensor:
                safe = torch.where(scale == 0, torch.ones_like(scale), scale)
                return (a * (127.0 / safe)).round().clamp(-127, 127).to(torch.int8)

            return quant(r[:, None]), quant(c[None, :]), r, c

        def dynamic(x: torch.Tensor = x) -> tuple:
            qr, _ = torch_npu.npu_dynamic_quant(x)
            qc, _ = torch_npu.npu_dynamic_quant(x.t().contiguous())
            absolute = x.float().abs()
            return qr, qc.t().contiguous(), absolute.amax(dim=1), absolute.amax(dim=0)

        record = {"case_name": case["case_name"], "candidates": {}}
        for name, fn in {"fp32_absmax_round_cast": composition, "two_dynamic_quant_transpose": dynamic}.items():
            try:
                out = fn()
                torch.npu.synchronize()
                verify.validate_ownership(inputs, out)
                accuracy = verify.compare(case, arrays, out, golden.calculate(arrays, {}))
                del out
            except (AssertionError, RuntimeError) as exc:
                record["candidates"][name] = {"status": "excluded", "reason": str(exc)}
                continue
            record["candidates"][name] = dict(status="qualified", accuracy=accuracy, **benchmark.measure(fn, torch.npu))
        qualified = {name: value for name, value in record["candidates"].items() if value["status"] == "qualified"}
        assert qualified
        record["selected"] = min(qualified, key=lambda name: qualified[name]["wall_us_median"])
        records.append(record)
        print(json.dumps(record), flush=True)
        args.report.write_text(
            json.dumps({"status": "running", "protocol": benchmark.PROTOCOL, "cases": records}, indent=2) + "\n"
        )
    args.report.write_text(
        json.dumps({"status": "passed", "protocol": benchmark.PROTOCOL, "cases": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
