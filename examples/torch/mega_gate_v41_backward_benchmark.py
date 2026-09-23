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
"""Compare V4.1 golden backward with HyperMegaGate on 2K through 32K tokens."""

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
from typing import Callable

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
import torch_npu  # noqa: F401  pylint: disable=unused-import
from torch_npu.profiler import ProfilerActivity, profile

from hyper_parallel.core.multicore import profiler as multicore_profiler
from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
# pylint: enable=wrong-import-position


_TOKENS = (2048, 4096, 8192, 16384, 32768)
_HIDDEN_SIZE = 5120
_EXPERT_COUNT = 384
_TOP_K = 6
_ROUTED_SCALING_FACTOR = 1.5
_MODES = ("weights-only", "combined")
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
    return (torch.arange(tokens, device=device) < vision_tokens).reshape(1, tokens)


def _golden_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the released V4.1 Projection and Route graph."""
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = functional.linear(flattened.float(), weight.float())  # pylint: disable=not-callable
    route_scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    correction_bias = text_bias
    if image_mask is not None:
        correction_bias = torch.where(
            image_mask.reshape(-1, 1),
            vision_bias.unsqueeze(0),
            text_bias.unsqueeze(0),
        )
    expert_indices = torch.topk(route_scores + correction_bias, _TOP_K, dim=-1, sorted=False).indices
    selected_scores = route_scores.gather(1, expert_indices)
    denominator = selected_scores.sum(dim=-1, keepdim=True) + 1.0e-20
    routing_weights = selected_scores / denominator * _ROUTED_SCALING_FACTOR
    return logits, routing_weights, expert_indices


def _make_backward_call(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, torch.Tensor],
    logits_grad: torch.Tensor,
    routing_weights_grad: torch.Tensor,
    mode: str,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Create a retained-graph backward call for one output-gradient mode."""
    def backward_call() -> tuple[torch.Tensor, ...]:
        """Differentiate the retained graph without accumulating leaf gradients."""
        return _differentiate(
            outputs,
            inputs,
            logits_grad,
            routing_weights_grad,
            mode,
            retain_graph=True,
        )

    return backward_call


def _differentiate(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, torch.Tensor],
    logits_grad: torch.Tensor,
    routing_weights_grad: torch.Tensor,
    mode: str,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    """Differentiate one Gate graph without accumulating leaf gradients."""
    logits, routing_weights, _ = outputs
    if mode == "weights-only":
        selected_outputs = (routing_weights,)
        output_grads = (routing_weights_grad,)
    elif mode == "combined":
        selected_outputs = (logits, routing_weights)
        output_grads = (logits_grad, routing_weights_grad)
    else:
        raise ValueError(f"unsupported backward mode: {mode}")

    return torch.autograd.grad(
        selected_outputs,
        inputs,
        grad_outputs=output_grads,
        retain_graph=retain_graph,
    )


def _measure_host(
    call: Callable[[], tuple[torch.Tensor, ...]],
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    """Measure synchronized single-call latency and stable repeated-call throughput."""
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


def _profile_backward(
    call: Callable[[], tuple[torch.Tensor, ...]],
    warmup: int,
    iterations: int,
    trace_path: Path | None,
) -> dict[str, object]:
    """Measure device time and kernel composition for backward only."""
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
    if not rows or any("Name" not in row or "Duration(us)" not in row for row in rows):
        raise RuntimeError(f"invalid profiler kernel data: {csv_path}")

    kernel_us: dict[str, float] = {}
    kernel_calls: Counter[str] = Counter()
    for row in rows:
        name = row["Name"]
        kernel_us[name] = kernel_us.get(name, 0.0) + float(row["Duration(us)"])
        kernel_calls[name] += 1
    device_us = sum(kernel_us.values())
    top_kernels = sorted(kernel_us.items(), key=lambda item: item[1], reverse=True)[:10]
    matmul_implementation_calls = Counter()
    matmul_block_calls = Counter()
    for row in rows:
        implementation = next(
            (name for name in ("MatMulV2", "MatMulV3") if name in row["Name"]),
            None,
        )
        if implementation is None:
            continue
        matmul_implementation_calls[implementation] += 1
        matmul_block_calls[f"{implementation}:{int(row['Block Num'])}"] += 1
    return {
        "device_ms": device_us / (1000.0 * iterations),
        "kernels_per_call": len(rows) / iterations,
        "kernel_calls_per_call": {
            name: count / iterations for name, count in sorted(kernel_calls.items())
        },
        "top_kernels_ms_per_call": {
            name: duration / (1000.0 * iterations) for name, duration in top_kernels
        },
        "matmul_implementation_calls_per_call": {
            name: count / iterations for name, count in sorted(matmul_implementation_calls.items())
        },
        "matmul_block_calls_per_call": {
            name: count / iterations for name, count in sorted(matmul_block_calls.items())
        },
    }


def _performance_errors(
    mode: str,
    include_forward: bool,
    route_mode: str,
    golden_before_device: dict[str, object],
    candidate_device: dict[str, object],
    golden_after_device: dict[str, object],
    golden_before_host: dict[str, object],
    candidate_host: dict[str, object],
    golden_after_host: dict[str, object],
) -> tuple[list[str], list[str], float, float]:
    """Apply the forward benchmark's path, drift, and latency gates."""
    device_errors = []
    expected_golden_kernels = (20.0 if mode == "weights-only" else 21.0) + (16.0 if include_forward else 0.0)
    expected_candidate_kernels = (11.0 if mode == "weights-only" else 12.0) + (4.0 if include_forward else 0.0)
    expected_matmuls = 3.0 if include_forward else 2.0
    for label, measurement, expected_kernels in (
        ("golden_before", golden_before_device, expected_golden_kernels),
        ("candidate", candidate_device, expected_candidate_kernels),
        ("golden_after", golden_after_device, expected_golden_kernels),
    ):
        check_kernel_count = not (
            label.startswith("golden") and include_forward and route_mode != "text"
        )
        if check_kernel_count and measurement["kernels_per_call"] != expected_kernels:
            device_errors.append(
                f"{label} expected {expected_kernels:.0f} backward kernels, "
                f"got {measurement['kernels_per_call']}"
            )
        if sum(measurement["matmul_implementation_calls_per_call"].values()) != expected_matmuls:
            device_errors.append(
                f"{label} expected {expected_matmuls:.0f} projection Matmuls, "
                f"got {measurement['matmul_implementation_calls_per_call']}"
            )

    matmul_implementations = (
        golden_before_device["matmul_implementation_calls_per_call"],
        candidate_device["matmul_implementation_calls_per_call"],
        golden_after_device["matmul_implementation_calls_per_call"],
    )
    if not matmul_implementations[0] == matmul_implementations[1] == matmul_implementations[2]:
        device_errors.append(f"candidate Matmul selection differs from golden: {matmul_implementations}")
    matmul_blocks = (
        golden_before_device["matmul_block_calls_per_call"],
        candidate_device["matmul_block_calls_per_call"],
        golden_after_device["matmul_block_calls_per_call"],
    )
    if not matmul_blocks[0] == matmul_blocks[1] == matmul_blocks[2]:
        device_errors.append(f"candidate Matmul block count differs from golden: {matmul_blocks}")

    golden_device_ms = min(golden_before_device["device_ms"], golden_after_device["device_ms"])
    device_drift_percent = (
        100.0 * abs(golden_after_device["device_ms"] - golden_before_device["device_ms"])
        / golden_device_ms
    )
    if device_drift_percent > 10.0:
        device_errors.append(f"golden device-time drift exceeds 10%: {device_drift_percent:.1f}%")
    if candidate_device["device_ms"] >= golden_device_ms:
        device_errors.append(
            f"candidate device time {candidate_device['device_ms']:.6f} ms "
            f">= golden {golden_device_ms:.6f} ms"
        )

    golden_host_ms = min(
        golden_before_host["steady_median_ms_per_call"],
        golden_after_host["steady_median_ms_per_call"],
    )
    golden_single_ms = min(
        golden_before_host["single_call_median_ms"],
        golden_after_host["single_call_median_ms"],
    )
    host_drift_percent = 100.0 * abs(
        golden_after_host["steady_median_ms_per_call"]
        - golden_before_host["steady_median_ms_per_call"]
    ) / golden_host_ms
    host_errors = []
    if host_drift_percent > 10.0:
        host_errors.append(f"golden host-time drift exceeds 10%: {host_drift_percent:.1f}%")
    if candidate_host["steady_median_ms_per_call"] >= golden_host_ms:
        host_errors.append(
            f"candidate steady host time {candidate_host['steady_median_ms_per_call']:.6f} ms "
            f">= golden {golden_host_ms:.6f} ms"
        )
    if candidate_host["single_call_median_ms"] >= golden_single_ms:
        host_errors.append(
            f"candidate single-call host time {candidate_host['single_call_median_ms']:.6f} ms "
            f">= golden {golden_single_ms:.6f} ms"
        )
    return device_errors, host_errors, device_drift_percent, host_drift_percent


def _export_mega_kernel_trace(
    call: Callable[[], tuple[torch.Tensor, ...]],
    trace_path: Path,
) -> dict[str, object]:
    """Export one profiled RouteGrad invocation and return its trace metadata."""
    with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
        call()
        torch.npu.synchronize()
        profiler.step()
    trace = profiler.export_chrome_trace(trace_path)
    metadata = trace["megaKernelCycleTrace"]
    return {
        "path": str(trace_path),
        "invocation_count": metadata["invocationCount"],
        "record_count": metadata["recordCount"],
        "dropped_record_count": metadata["droppedRecordCount"],
        "warnings": metadata["warnings"],
    }


def _compare_gradients(
    golden_call: Callable[[], tuple[torch.Tensor, ...]],
    candidate_call: Callable[[], tuple[torch.Tensor, ...]],
) -> dict[str, object]:
    """Check BF16 hidden and weight gradients before timing."""
    golden_grads = golden_call()
    # Golden and candidate are independent retained graphs and are timed in
    # separately synchronized windows below.  Use the same isolation for the
    # accuracy check instead of interleaving both backends in one task queue.
    torch.npu.synchronize()
    candidate_grads = candidate_call()
    torch.npu.synchronize()
    finite = all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in (*golden_grads, *candidate_grads)
    )
    if not finite:
        nonfinite_counts = [
            int((~torch.isfinite(gradient)).sum().item())
            for gradient in (*golden_grads, *candidate_grads)
        ]
        return {
            "passed": False,
            "nonfinite_counts": nonfinite_counts,
            "reason": "benchmark inputs produced non-finite golden or candidate gradients",
        }
    max_abs = [
        float((actual.float() - expected.float()).abs().max().item())
        for actual, expected in zip(candidate_grads, golden_grads)
    ]
    try:
        for actual, expected in zip(candidate_grads, golden_grads):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=1.0e-2, atol=1.0e-4)
    except AssertionError as error:
        return {"passed": False, "max_abs": max_abs, "reason": str(error)[:500]}
    return {"passed": True, "max_abs": max_abs}


def _make_inputs(
    tokens: int,
    device: torch.device,
    route_mode: str,
    vision_token_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Create fixed BF16 model inputs, two biases, and an optional visual mask."""
    hidden_states = torch.randn((1, tokens, _HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    weight = torch.randn((_EXPERT_COUNT, _HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    # Match the model-scale initialization used by the backward parity matrix;
    # unit-variance weights make FP32 Softplus underflow before Sqrt backward.
    weight.mul_(0.02)
    text_bias = torch.empty((_EXPERT_COUNT,), device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    vision_bias = torch.empty((_EXPERT_COUNT,), device=device, dtype=torch.float32).uniform_(-0.5, 0.5)
    image_mask = _make_image_mask(tokens, device, route_mode, vision_token_ratio)
    return hidden_states, weight, text_bias, vision_bias, image_mask


def _build_backend(
    backend: str,
    gate: MegaGate,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor | None,
    logits_grad: torch.Tensor,
    routing_weights_grad: torch.Tensor,
    mode: str,
    include_forward: bool,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Build a backward-only or complete forward-backward call."""
    hidden_leaf = hidden_states.detach().clone().requires_grad_(True)
    weight_leaf = weight.detach().clone().requires_grad_(True)
    if backend == "golden":
        gradient_inputs = (hidden_leaf, weight_leaf)

        def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Build one golden forward graph."""
            return _golden_forward(hidden_leaf, weight_leaf, text_bias, vision_bias, image_mask)
    elif backend == "candidate":
        gate.weight = torch.nn.Parameter(weight_leaf)
        gate.bias = torch.nn.Parameter(text_bias, requires_grad=False)
        if gate.bias_vl is not None:
            gate.bias_vl = torch.nn.Parameter(vision_bias, requires_grad=False)
        gradient_inputs = (hidden_leaf, gate.weight)

        def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Build one MegaGate forward graph."""
            return gate(hidden_leaf, image_mask=image_mask)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    if include_forward:
        def forward_backward_call() -> tuple[torch.Tensor, ...]:
            """Build and differentiate a fresh graph for full training latency."""
            return _differentiate(
                forward(),
                gradient_inputs,
                logits_grad,
                routing_weights_grad,
                mode,
                retain_graph=False,
            )

        return forward_backward_call
    outputs = forward()
    return _make_backward_call(
        outputs,
        gradient_inputs,
        logits_grad,
        routing_weights_grad,
        mode,
    )


def _run_case(
    tokens: int,
    mode: str,
    device: torch.device,
    warmup: int,
    profile_iterations: int,
    wall_iterations: int,
    wall_repeats: int,
    trace_dir: Path | None,
    mega_kernel_trace_dir: Path | None,
    include_forward: bool,
    route_mode: str,
    vision_token_ratio: float,
) -> dict[str, object]:
    """Measure one token count and gradient mode with bracketed golden samples."""
    hidden_states, weight, text_bias, vision_bias, image_mask = _make_inputs(
        tokens, device, route_mode, vision_token_ratio,
    )
    gate = MegaGate(
        hidden_size=_HIDDEN_SIZE,
        num_experts=_EXPERT_COUNT,
        scoring_func="sqrtsoftplus",
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        vision_enabled=route_mode != "text",
    )
    logits_grad = torch.randn((tokens, _EXPERT_COUNT), device=device, dtype=torch.float32)
    routing_weights_grad = torch.randn((tokens, _TOP_K), device=device, dtype=torch.float32)

    golden_call = _build_backend(
        "golden", gate, hidden_states, weight, text_bias, vision_bias, image_mask,
        logits_grad, routing_weights_grad, mode, include_forward,
    )
    candidate_call = _build_backend(
        "candidate", gate, hidden_states, weight, text_bias, vision_bias, image_mask,
        logits_grad, routing_weights_grad, mode, include_forward,
    )
    accuracy = _compare_gradients(golden_call, candidate_call)
    result: dict[str, object] = {
        "tokens": tokens,
        "mode": mode,
        "route_mode": route_mode,
        "vision_tokens": 0 if image_mask is None else int(image_mask.sum().item()),
        "accuracy": accuracy,
    }
    if not accuracy["passed"]:
        result["passed"] = False
        return result

    golden_before_host = _measure_host(golden_call, warmup, wall_iterations, wall_repeats)
    candidate_host = _measure_host(candidate_call, warmup, wall_iterations, wall_repeats)
    golden_after_host = _measure_host(golden_call, warmup, wall_iterations, wall_repeats)
    golden_before_device = _profile_backward(
        golden_call, warmup, profile_iterations,
        trace_dir / f"t{tokens}_{mode}_golden_before.json" if trace_dir else None,
    )
    candidate_device = _profile_backward(
        candidate_call, warmup, profile_iterations,
        trace_dir / f"t{tokens}_{mode}_candidate.json" if trace_dir else None,
    )
    golden_after_device = _profile_backward(
        golden_call, warmup, profile_iterations,
        trace_dir / f"t{tokens}_{mode}_golden_after.json" if trace_dir else None,
    )
    device_errors, host_errors, device_drift_percent, host_drift_percent = _performance_errors(
        mode,
        include_forward,
        route_mode,
        golden_before_device,
        candidate_device,
        golden_after_device,
        golden_before_host,
        candidate_host,
        golden_after_host,
    )
    golden_host_ms = min(
        golden_before_host["steady_median_ms_per_call"],
        golden_after_host["steady_median_ms_per_call"],
    )
    golden_device_ms = min(golden_before_device["device_ms"], golden_after_device["device_ms"])
    mega_kernel_trace = None
    if mega_kernel_trace_dir is not None:
        mega_kernel_trace = _export_mega_kernel_trace(
            candidate_call,
            mega_kernel_trace_dir / f"t{tokens}_{mode}_mega_kernel.json",
        )
    result.update(
        golden_before_host=golden_before_host,
        candidate_host=candidate_host,
        golden_after_host=golden_after_host,
        golden_before_device=golden_before_device,
        candidate_device=candidate_device,
        golden_after_device=golden_after_device,
        host_speedup=golden_host_ms / candidate_host["steady_median_ms_per_call"],
        device_speedup=golden_device_ms / candidate_device["device_ms"],
        golden_device_drift_percent=device_drift_percent,
        golden_host_drift_percent=host_drift_percent,
        path_errors=device_errors,
        host_errors=host_errors,
        mega_kernel_trace=mega_kernel_trace,
        passed=not device_errors and not host_errors,
    )
    return result


def main() -> int:
    """Run the V4.1 backward comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--mk-trace-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", "--profile-iterations", dest="iterations", type=int, default=20)
    parser.add_argument("--wall-iterations", type=int, default=500)
    parser.add_argument("--wall-repeats", type=int, default=5)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--route-mode", choices=_ROUTE_MODES, default="vision-mixed")
    parser.add_argument("--vision-token-ratio", type=float, default=0.25)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tokens", type=int, nargs="+", choices=_TOKENS, default=None)
    selection.add_argument("--quick", action="store_true", help="measure only the 2K combined case")
    parser.add_argument(
        "--include-forward",
        action="store_true",
        help="measure a newly built forward graph and its backward in every call",
    )
    args = parser.parse_args()
    if min(args.iterations, args.wall_iterations, args.wall_repeats) < 1 or args.warmup < 0:
        parser.error("warmup must be nonnegative and all iteration counts must be positive")
    if args.route_mode == "vision-mixed" and not 0.0 < args.vision_token_ratio < 1.0:
        parser.error("vision-token-ratio must be between zero and one for vision-mixed")
    if not torch.npu.is_available():
        raise RuntimeError("the V4.1 backward benchmark requires an Ascend NPU")

    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    torch.manual_seed(18)
    torch.npu.manual_seed(18)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    if args.mk_trace_dir is not None:
        args.mk_trace_dir.mkdir(parents=True, exist_ok=True)
    cases = ((2048, "combined"),) if args.quick else (
        (tokens, mode) for tokens in (args.tokens or _TOKENS) for mode in _MODES
    )
    output: dict[str, object] = {
        "scope": (
            "V4.1 forward and backward: golden autograd vs MegaGate"
            if args.include_forward
            else "V4.1 backward only: golden autograd vs MegaGate"
        ),
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
        "include_forward": args.include_forward,
        "route_mode": args.route_mode,
        "vision_token_ratio": args.vision_token_ratio,
        "cases": [],
    }
    exit_code = 0
    for tokens, mode in cases:
        result = _run_case(
            tokens,
            mode,
            device,
            args.warmup,
            args.iterations,
            args.wall_iterations,
            args.wall_repeats,
            args.trace_dir,
            args.mk_trace_dir,
            args.include_forward,
            args.route_mode,
            args.vision_token_ratio,
        )
        output["cases"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        if not result["accuracy"]["passed"]:
            exit_code = 1
            print(f"T={tokens} mode={mode}: accuracy failed: {result['accuracy']}", flush=True)
            continue
        if not result["passed"]:
            exit_code = 1
        print(
            f"T={tokens} mode={mode}: "
            f"host={result['candidate_host']['steady_median_ms_per_call']:.4f}ms "
            f"({result['host_speedup']:.3f}x), "
            f"device={result['candidate_device']['device_ms']:.4f}ms "
            f"({result['device_speedup']:.3f}x), "
            f"kernels={result['candidate_device']['kernels_per_call']:.1f}, "
            f"passed={result['passed']}",
            flush=True,
        )
        for error in (*result["path_errors"], *result["host_errors"]):
            print(f"  ERROR: {error}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
