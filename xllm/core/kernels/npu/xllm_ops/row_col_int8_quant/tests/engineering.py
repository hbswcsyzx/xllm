# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Task-defined cross-entry, storage, stream and memory acceptance checks."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu

OP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP))
import solution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.public.resolve()))
    import generate_data
    import golden
    import verify

    cases = json.loads((args.public / "case.json").read_text())
    coverage = json.loads((args.public / "coverage_requirements.json").read_text())
    by_name = {case["case_name"]: case for case in cases}
    torch.npu.set_device(0)
    report = {"status": "running", "records": []}
    records = report["records"]

    def record(kind: str, name: str, entry: str, **detail: Any) -> None:
        item = dict(kind=kind, case_name=name, entry=entry, status="passed", **detail)
        records.append(item)
        print(json.dumps(item), flush=True)

    def check(case: dict, arrays: list, inputs: list, outputs: tuple) -> list:
        torch.npu.synchronize()
        verify.validate_ownership(inputs, outputs)
        result = verify.compare(case, arrays, outputs, golden.calculate(arrays, {}))
        assert np.array_equal(verify.cpu_array(inputs[0]).view(np.uint16), arrays[0].view(np.uint16))
        return result

    def rejection(fn: Callable, value: Any) -> str:
        try:
            fn(*value)
        except (ValueError, TypeError, RuntimeError) as exc:
            return str(exc)
        raise AssertionError("invalid metadata was accepted")

    try:
        for case in cases:
            arrays = generate_data.generate(case)
            inputs = verify.device_inputs(case, arrays, "npu:0")
            out = solution.run_direct(inputs, {})
            record("public", case["case_name"], "run_direct", checks=check(case, arrays, inputs, out))
            del inputs, out
        for name in coverage["extra_input_case_ids"]:
            case = copy.deepcopy(by_name[name.split("__extra_seed_")[0]])
            case.update(case_name=name, kind="random", performance=False)
            arrays = generate_data.generate(case)
            storage = torch.empty(arrays[0].size + 16, dtype=torch.float16, device="npu:0")
            x = storage[1 : arrays[0].size + 1].view(arrays[0].shape)
            x.copy_(torch.from_numpy(arrays[0].view(np.int16).copy()).to("npu:0").view(torch.float16))
            for entry in coverage["extra_entries"]:
                out = getattr(solution, entry)([x], {})
                record("extra", name, entry, storage_offset=x.storage_offset(), checks=check(case, arrays, [x], out))
                del out
            del x, storage
        base = torch.empty((257, 1026), dtype=torch.float16, device="npu:0")
        invalid = {
            "attributes_type": ([base], []),
            "unknown_attribute": ([base], {"unknown": 1}),
            "input_count": ([], {}),
            "cpu_input": ([torch.empty((257, 1026), dtype=torch.float16)], {}),
            "bf16": ([base.to(torch.bfloat16)], {}),
            "float32": ([base.float()], {}),
            "rank": ([base.flatten()[:8].view(2, 2, 2)], {}),
            "noncontiguous": (
                [torch.empty((257, 2052), dtype=torch.float16, device="npu:0")[:, ::2]],
                {},
            ),
        }
        for name, shape in {
            "empty_m": (0, 1026),
            "empty_n": (257, 0),
            "large_m": (4097, 1026),
            "large_n": (257, 4098),
            "odd_n": (257, 3),
        }.items():
            invalid[name] = ([torch.empty(shape, dtype=torch.float16, device="npu:0")], {})
        assert invalid["rank"][0][0].shape == (2, 2, 2)
        noncontiguous = invalid["noncontiguous"][0][0]
        assert noncontiguous.shape == base.shape and not noncontiguous.is_contiguous()
        for name in coverage["negative_scenarios"]:
            metadata = [
                {
                    "shape": list(value.shape),
                    "strides": list(value.stride()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                for value in invalid[name][0]
            ]
            for entry in coverage["negative_entries"]:
                record(
                    "negative",
                    name,
                    entry,
                    error=rejection(getattr(solution, entry), invalid[name]),
                    input_metadata=metadata,
                    attributes_type=type(invalid[name][1]).__name__,
                )
        for name in ("bf16", "float32", "rank", "noncontiguous", "empty_m", "odd_n"):
            record("metadata", name, "native", error=rejection(solution._NATIVE.run, (invalid[name][0][0],)))
        del invalid, base

        streams = [torch.npu.Stream() for _ in range(2)]
        pending = []
        for stream, name in zip(streams, coverage["stream_case_ids"]):
            case = by_name[name]
            arrays = generate_data.generate(case)
            with torch.npu.stream(stream):
                inputs = verify.device_inputs(case, arrays, "npu:0")
                solution.prepare(inputs, {})
                outputs = [solution.run(inputs, {}) for _ in range(coverage["min_prepared_calls_per_stream"])]
            pending.append((case, arrays, inputs, outputs))
        for case, arrays, inputs, outputs in pending:
            for out in outputs:
                check(case, arrays, inputs, out)
            record("stream", case["case_name"], "run", calls=len(outputs))
        del pending, outputs, inputs, out

        case = by_name[coverage["stream_case_ids"][0]]
        arrays = generate_data.generate(case)
        for entry in ("run", "run_direct"):
            with torch.npu.stream(streams[0]):
                inputs = verify.device_inputs(case, arrays, "npu:0")
                outputs = getattr(solution, entry)(inputs, {})
                del inputs
            with torch.npu.stream(streams[1]):
                pressure = [
                    torch.empty(arrays[0].shape, dtype=torch.float16, device="npu:0").fill_(17) for _ in range(32)
                ]
            torch.npu.synchronize()
            verify.compare(case, arrays, outputs, golden.calculate(arrays, {}))
            record("input_early_release", case["case_name"], entry, pressure_allocations=32)
            del pressure, outputs

        for entry in ("run", "run_direct"):
            inputs = verify.device_inputs(case, arrays, "npu:0")
            retained = [getattr(solution, entry)(inputs, {}) for _ in range(8)]
            for _ in range(32):
                temporary = getattr(solution, entry)(inputs, {})
                del temporary
                pressure = torch.empty(arrays[0].shape, dtype=torch.int8, device="npu:0").fill_(23)
                del pressure
            pointers = [t.untyped_storage().data_ptr() for group in retained for t in group]
            assert len(set(pointers)) == len(pointers)
            for out in retained:
                check(case, arrays, inputs, out)
            record("output_lifetime", case["case_name"], entry, retained=8, released=32)
            del retained, out, inputs

        for case in (case for case in cases if case["performance"]):
            arrays = generate_data.generate(case)
            inputs = verify.device_inputs(case, arrays, "npu:0")
            for entry in ("run", "run_direct"):
                fn = getattr(solution, entry)
                for _ in range(50):
                    out = fn(inputs, {})
                    del out
                torch.npu.synchronize()
                before = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                out = fn(inputs, {})
                torch.npu.synchronize()
                peak = torch.npu.max_memory_allocated() - before
                output_bytes = sum(t.numel() * t.element_size() for t in out)
                m, n = inputs[0].shape
                workspace_limit = 16 * 1024 * 1024 + 4 * ((m + 63) // 64) * ((n + 255) // 256) * (64 + 256)
                actual_workspace = solution._NATIVE.workspace_size()
                assert actual_workspace <= workspace_limit
                assert peak <= output_bytes + workspace_limit + 16 * 1024
                del out
                torch.npu.synchronize()
                for _ in range(20):
                    out = fn(inputs, {})
                    del out
                torch.npu.synchronize()
                after = torch.npu.memory_allocated()
                assert after == before, (before, after)
                record(
                    "memory",
                    case["case_name"],
                    entry,
                    peak_increment=peak,
                    output_bytes=output_bytes,
                    workspace_bytes=actual_workspace,
                    workspace_limit=workspace_limit,
                    active_before=before,
                    active_after=after,
                    repeats=20,
                )
            del inputs

        case = cases[3]
        arrays = generate_data.generate(case)
        inputs = verify.device_inputs(case, arrays, "npu:0")
        out = solution.run(inputs, {})
        for name in coverage["verifier_checks"]:
            bad = list(out)
            if name.startswith("CPU"):
                bad[0] = bad[0].cpu()
                function = lambda bad=bad: verify.compare(case, arrays, bad, golden.calculate(arrays, {}))
            elif name.startswith("input"):
                bad[0] = inputs[0].view(torch.int8)
                function = lambda bad=bad: verify.validate_ownership(inputs, bad)
            else:
                parent = torch.empty(out[0].numel() + 32, dtype=torch.int8, device="npu:0")
                bad[0] = parent[: out[0].numel()].view(out[0].shape)
                bad[1] = parent[1 : out[0].numel() + 1].view(out[0].shape)
                function = lambda bad=bad: verify.validate_ownership(inputs, bad)
            try:
                function()
            except AssertionError as exc:
                record("verifier", name, "official", error=str(exc))
            else:
                raise AssertionError("verifier accepted counterexample")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
