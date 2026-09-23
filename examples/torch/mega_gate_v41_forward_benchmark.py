# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compare V4.1 golden forward with HyperMegaGate on measured shapes."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import median

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
import torch_npu  # noqa: F401  pylint: disable=unused-import
from torch_npu.profiler import ProfilerActivity, profile

from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
# pylint: enable=wrong-import-position


_TOKENS = (2048, 4096, 8192, 16384, 32768)
_HIDDEN = 5120
_EXPERTS = 384
_TOP_K = 6
_SCALING = 1.5
_ROUTE_MODES = ("vision-mixed", "vision", "text")


def _make_image_mask(
    tokens: int,
    device: torch.device,
    route_mode: str,
    vision_token_ratio: float,
) -> torch.Tensor | None:
    """Create a deterministic contiguous visual-token block for profiling."""
    if route_mode == "text":
        return None
    if route_mode == "vision":
        vision_tokens = tokens
    else:
        vision_tokens = max(1, min(tokens - 1, round(tokens * vision_token_ratio)))
    positions = torch.arange(tokens, device=device)
    return (positions < vision_tokens).reshape(1, tokens)


def _golden_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the released V4.1 Route chain, including both input casts."""
    logits = functional.linear(hidden.reshape(-1, _HIDDEN).float(), weight.float())  # pylint: disable=not-callable
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    correction_bias = text_bias
    if image_mask is not None:
        correction_bias = torch.where(
            image_mask.reshape(-1, 1),
            vision_bias.unsqueeze(0),
            text_bias.unsqueeze(0),
        )
    indices = torch.topk(scores + correction_bias, _TOP_K, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * _SCALING, indices


def _candidate_forward(
    gate: MegaGate,
    hidden: torch.Tensor,
    image_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the formal module with internal casts, Matmul, and scheduled Route."""
    return gate(hidden, image_mask=image_mask)


def _check_accuracy(
    golden: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    """Match logits and weights by expert ID rather than TopK order."""
    golden_logits, golden_weights, golden_indices = golden
    candidate_logits, candidate_weights, candidate_indices = candidate
    golden_order = golden_indices.argsort(dim=-1)
    candidate_order = candidate_indices.argsort(dim=-1)
    golden_indices = golden_indices.gather(1, golden_order)
    candidate_indices = candidate_indices.gather(1, candidate_order)
    golden_weights = golden_weights.gather(1, golden_order)
    candidate_weights = candidate_weights.gather(1, candidate_order)
    mismatched_tokens = int((candidate_indices != golden_indices).any(dim=1).sum().item())
    result: dict[str, object] = {
        "logits_max_abs": float((candidate_logits - golden_logits).abs().max().item()),
        "weights_max_abs": float((candidate_weights - golden_weights).abs().max().item()),
        "expert_set_mismatch_tokens": mismatched_tokens,
    }
    try:
        torch.testing.assert_close(candidate_logits, golden_logits, rtol=1.0e-5, atol=1.0e-6)
        if mismatched_tokens:
            raise AssertionError(f"selected expert sets differ for {mismatched_tokens} tokens")
        torch.testing.assert_close(candidate_weights, golden_weights, rtol=1.0e-5, atol=1.0e-6)
    except AssertionError as error:
        result.update(passed=False, reason=str(error)[:500])
    else:
        result["passed"] = True
    return result


def _measure_wall(call, warmup: int, iterations: int, repeats: int) -> dict[str, object]:
    """Measure output-ready latency and repeated-call throughput without profiler overhead."""
    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    single_call_ms = []
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        single_call_ms.append((time.perf_counter_ns() - start_ns) / 1.0e6)

    steady_ms_per_call = []
    enqueue_ms_per_call = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start_ns = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        enqueue_end_ns = time.perf_counter_ns()
        torch.npu.synchronize()
        end_ns = time.perf_counter_ns()
        enqueue_ms_per_call.append((enqueue_end_ns - start_ns) / (iterations * 1.0e6))
        steady_ms_per_call.append((end_ns - start_ns) / (iterations * 1.0e6))

    ordered_single_ms = sorted(single_call_ms)
    p90_index = (9 * iterations + 9) // 10 - 1
    return {
        "single_call_median_ms": median(single_call_ms),
        "single_call_p90_ms": ordered_single_ms[p90_index],
        "steady_median_ms_per_call": median(steady_ms_per_call),
        "enqueue_median_ms_per_call": median(enqueue_ms_per_call),
        "steady_repeats_ms_per_call": steady_ms_per_call,
        "single_call_samples_ms": single_call_ms,
    }


def _profile(
    call,
    warmup: int,
    iterations: int,
    trace_path: Path | None,
) -> dict[str, object]:
    """Record device kernels and upper-level ACLNN calls per invocation."""
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as profiler:
        for _ in range(iterations):
            call()
        torch.npu.synchronize()
    if trace_path is not None:
        profiler.export_chrome_trace(str(trace_path))
    profiler.prof_if.analyse()
    csv_path = Path(profiler.prof_if.prof_path) / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any("Duration(us)" not in row or "Name" not in row for row in rows):
        raise RuntimeError(f"invalid profiler kernel_details.csv: {csv_path}")
    operator_path = csv_path.with_name("operator_details.csv")
    with operator_path.open(newline="", encoding="utf-8") as handle:
        operator_rows = list(csv.DictReader(handle))
    aclnn_calls = Counter(row["Name"] for row in operator_rows if row.get("Name", "").startswith("aclnn"))
    aclnn_host_us = sum(
        float(row["Host Self Duration(us)"])
        for row in operator_rows if row.get("Name", "").startswith("aclnn")
    )
    calls = Counter(row["Name"] for row in rows)
    durations: dict[str, float] = {}
    for row in rows:
        name = row["Name"]
        durations[name] = durations.get(name, 0.0) + float(row["Duration(us)"])
    kernel_ms = {name: duration / (1000.0 * iterations) for name, duration in durations.items()}
    input_cast_ms = sum(
        duration for name, duration in kernel_ms.items()
        if "Cast" in name
    )
    matmul_ms = sum(
        duration for name, duration in kernel_ms.items() if "MatMulV2" in name or "MatMulV3" in name
    )
    matmul_implementations = sorted({
        implementation
        for name in calls
        for implementation in ("MatMulV2", "MatMulV3")
        if implementation in name
    })
    device_ms = sum(kernel_ms.values())
    return {
        "device_ms": device_ms,
        "kernels_per_call": len(rows) / iterations,
        "kernel_calls_per_call": {name: count / iterations for name, count in calls.items()},
        "kernel_ms": kernel_ms,
        "kernel_phase_ms": {
            "input_cast": input_cast_ms,
            "matmul": matmul_ms,
            "route": device_ms - input_cast_ms - matmul_ms,
        },
        "aclnn_calls_per_call": {name: count / iterations for name, count in aclnn_calls.items()},
        "aclnn_host_us_per_call": aclnn_host_us / iterations,
        "matmul_implementations": matmul_implementations,
        "matmul_block_nums": sorted({
            int(row["Block Num"])
            for row in rows
            if "MatMulV2" in row["Name"] or "MatMulV3" in row["Name"]
        }),
    }


def _path_errors(
    golden_before: dict[str, object],
    candidate: dict[str, object],
    golden_after: dict[str, object],
    route_mode: str,
) -> list[str]:
    """Require candidate and golden to select the same single Matmul implementation."""
    errors = []
    for label, measurement in (
        ("golden_before", golden_before), ("candidate", candidate), ("golden_after", golden_after),
    ):
        calls = measurement["kernel_calls_per_call"]
        matmul_v2 = sum(count for name, count in calls.items() if "MatMulV2" in name)
        matmul_v3 = sum(count for name, count in calls.items() if "MatMulV3" in name)
        if matmul_v2 + matmul_v3 != 1.0:
            errors.append(f"{label} Projection is not exactly one Matmul: V2={matmul_v2}, V3={matmul_v3}")
        if label.startswith("golden") and route_mode == "text" and measurement["kernels_per_call"] != 16.0:
            errors.append(f"{label} expected 16 full-forward kernels, got {measurement['kernels_per_call']}")
    calls = candidate["kernel_calls_per_call"]
    route_calls = sum(count for name, count in calls.items() if "HyperMegaGateRoute" in name)
    cast_calls = sum(count for name, count in calls.items() if "Cast" in name)
    if route_calls != 1.0 or cast_calls != 2.0 or candidate["kernels_per_call"] != 4.0:
        errors.append(
            f"candidate expected two casts, one Matmul and one Route; "
            f"observed casts={cast_calls}, Route={route_calls}, kernels={candidate['kernels_per_call']}"
        )
    selected_implementations = (
        golden_before["matmul_implementations"],
        candidate["matmul_implementations"],
        golden_after["matmul_implementations"],
    )
    if not selected_implementations[0] == selected_implementations[1] == selected_implementations[2]:
        errors.append(f"candidate Matmul selection differs from golden: {selected_implementations}")
    if candidate["matmul_block_nums"] != golden_before["matmul_block_nums"] or (
        candidate["matmul_block_nums"] != golden_after["matmul_block_nums"]
    ):
        errors.append(
            "candidate Matmul block count differs from golden: "
            f"{golden_before['matmul_block_nums']}, {candidate['matmul_block_nums']}, "
            f"{golden_after['matmul_block_nums']}"
        )
    aclnn_calls = candidate["aclnn_calls_per_call"]
    route_calls = aclnn_calls.get("aclnnHyperMegaGateRoute", 0.0)
    if route_calls != 1.0:
        errors.append(f"HyperMegaGateRoute expected one ACLNN call, observed {aclnn_calls}")
    return errors


def _run_shape(
    tokens: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    wall_iterations: int,
    wall_repeats: int,
    route_mode: str,
    vision_token_ratio: float,
    trace_dir: Path | None,
) -> dict[str, object]:
    """Check correctness first, then bracket the candidate profile with golden."""
    gate = MegaGate(
        hidden_size=_HIDDEN,
        num_experts=_EXPERTS,
        scoring_func="sqrtsoftplus",
        top_k=_TOP_K,
        routed_scaling_factor=_SCALING,
        vision_enabled=route_mode != "text",
    )
    hidden = torch.randn((1, tokens, _HIDDEN), device=device, dtype=torch.bfloat16)
    weight = torch.randn((_EXPERTS, _HIDDEN), device=device, dtype=torch.bfloat16)
    text_bias = torch.empty((_EXPERTS,), device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    vision_bias = torch.empty((_EXPERTS,), device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    image_mask = _make_image_mask(tokens, device, route_mode, vision_token_ratio)
    gate.weight = torch.nn.Parameter(weight, requires_grad=False)
    gate.bias = torch.nn.Parameter(text_bias, requires_grad=False)
    if gate.bias_vl is not None:
        gate.bias_vl = torch.nn.Parameter(vision_bias, requires_grad=False)

    def golden_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the golden path on the fixed operands."""
        return _golden_forward(hidden, weight, text_bias, vision_bias, image_mask)

    def candidate_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the candidate path on the fixed operands."""
        return _candidate_forward(gate, hidden, image_mask)

    with torch.no_grad():
        accuracy = _check_accuracy(golden_call(), candidate_call())
        result: dict[str, object] = {
            "tokens": tokens,
            "route_mode": route_mode,
            "vision_tokens": 0 if image_mask is None else int(image_mask.sum().item()),
            "accuracy": accuracy,
        }
        if not accuracy["passed"]:
            result["passed"] = False
            return result
        golden_before_wall = _measure_wall(golden_call, warmup, wall_iterations, wall_repeats)
        candidate_wall = _measure_wall(candidate_call, warmup, wall_iterations, wall_repeats)
        golden_after_wall = _measure_wall(golden_call, warmup, wall_iterations, wall_repeats)
        golden_before = _profile(
            golden_call,
            warmup,
            iterations,
            trace_dir / f"t{tokens}_golden_before.json" if trace_dir else None,
        )
        candidate = _profile(
            candidate_call,
            warmup,
            iterations,
            trace_dir / f"t{tokens}_candidate.json" if trace_dir else None,
        )
        golden_after = _profile(
            golden_call,
            warmup,
            iterations,
            trace_dir / f"t{tokens}_golden_after.json" if trace_dir else None,
        )
    errors = _path_errors(golden_before, candidate, golden_after, route_mode)
    baseline_ms = min(golden_before["device_ms"], golden_after["device_ms"])
    drift_percent = 100.0 * abs(golden_after["device_ms"] - golden_before["device_ms"]) / baseline_ms
    if drift_percent > 10.0:
        errors.append(f"golden device-time drift exceeds 10%: {drift_percent:.1f}%")
    if candidate["device_ms"] >= baseline_ms:
        errors.append(f"candidate device time {candidate['device_ms']:.6f} ms >= golden {baseline_ms:.6f} ms")
    wall_baseline_ms = min(
        golden_before_wall["steady_median_ms_per_call"], golden_after_wall["steady_median_ms_per_call"],
    )
    wall_single_baseline_ms = min(
        golden_before_wall["single_call_median_ms"], golden_after_wall["single_call_median_ms"],
    )
    wall_drift_percent = 100.0 * abs(
        golden_after_wall["steady_median_ms_per_call"] - golden_before_wall["steady_median_ms_per_call"]
    ) / wall_baseline_ms
    wall_errors = []
    if wall_drift_percent > 10.0:
        wall_errors.append(f"golden wall-time drift exceeds 10%: {wall_drift_percent:.1f}%")
    if candidate_wall["steady_median_ms_per_call"] >= wall_baseline_ms:
        wall_errors.append(
            f"candidate steady wall time {candidate_wall['steady_median_ms_per_call']:.6f} ms "
            f">= golden {wall_baseline_ms:.6f} ms"
        )
    if candidate_wall["single_call_median_ms"] >= wall_single_baseline_ms:
        wall_errors.append(
            f"candidate single-call wall time {candidate_wall['single_call_median_ms']:.6f} ms "
            f">= golden {wall_single_baseline_ms:.6f} ms"
        )
    result.update(
        golden_before=golden_before,
        candidate=candidate,
        golden_after=golden_after,
        golden_before_wall=golden_before_wall,
        candidate_wall=candidate_wall,
        golden_after_wall=golden_after_wall,
        golden_drift_percent=drift_percent,
        device_speedup=baseline_ms / candidate["device_ms"],
        wall_drift_percent=wall_drift_percent,
        wall_steady_speedup=wall_baseline_ms / candidate_wall["steady_median_ms_per_call"],
        wall_single_speedup=wall_single_baseline_ms / candidate_wall["single_call_median_ms"],
        wall_path_errors=wall_errors,
        path_errors=errors,
        passed=not errors and not wall_errors,
    )
    return result


def main() -> int:
    """Run the forward experiment on the common V4.1 2K, 4K, and 8K shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--wall-iterations", type=int, default=100)
    parser.add_argument("--wall-repeats", type=int, default=3)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--route-mode", choices=_ROUTE_MODES, default="vision-mixed")
    parser.add_argument("--vision-token-ratio", type=float, default=0.25)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tokens", type=int, nargs="+", choices=_TOKENS, default=None)
    selection.add_argument("--quick", action="store_true", help="profile only the 2K shape")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.wall_iterations < 1 or args.wall_repeats < 1:
        parser.error("warmup must be nonnegative; iterations, wall-iterations, and wall-repeats must be positive")
    if args.route_mode == "vision-mixed" and not 0.0 < args.vision_token_ratio < 1.0:
        parser.error("vision-token-ratio must be between zero and one for vision-mixed")
    if not torch.npu.is_available():
        raise RuntimeError("the V4.1 forward benchmark requires an Ascend NPU")
    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    torch.manual_seed(18)
    torch.npu.manual_seed(18)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, object] = {
        "scope": "V4.1 golden Route vs MegaGate fused text/vision Route",
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device_index),
            "mega_gate_module_source": str(Path(inspect.getfile(MegaGate)).resolve()),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "wall_iterations": args.wall_iterations,
        "wall_repeats": args.wall_repeats,
        "route_mode": args.route_mode,
        "vision_token_ratio": args.vision_token_ratio,
        "shapes": [],
    }
    for tokens in (args.tokens or (_TOKENS[:1] if args.quick else _TOKENS)):
        result = _run_shape(
            tokens,
            device,
            args.warmup,
            args.iterations,
            args.wall_iterations,
            args.wall_repeats,
            args.route_mode,
            args.vision_token_ratio,
            args.trace_dir,
        )
        output["shapes"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        if result["accuracy"]["passed"]:
            print(
                f"T={tokens}: golden={result['golden_before']['device_ms']:.4f}/"
                f"{result['golden_after']['device_ms']:.4f} ms, "
                f"candidate={result['candidate']['device_ms']:.4f} ms, "
                f"device_speedup={result['device_speedup']:.2f}x, "
                f"single_wall={result['candidate_wall']['single_call_median_ms']:.4f} ms, "
                f"steady_wall={result['candidate_wall']['steady_median_ms_per_call']:.4f} ms, "
                f"steady_speedup={result['wall_steady_speedup']:.2f}x, passed={result['passed']}",
                flush=True,
            )
        else:
            print(f"T={tokens}: accuracy failed: {result['accuracy']}", flush=True)
    return 0 if all(result["passed"] for result in output["shapes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
