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
"""Profile golden or HyperMegaGate V4.1 backward without forward kernels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
import torch_npu  # noqa: F401  pylint: disable=unused-import
from torch_npu.profiler import ProfilerActivity, profile
from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
# pylint: enable=wrong-import-position


_TOKENS = (2048, 4096, 8192)
_HIDDEN = 5120
_EXPERTS = 384
_TOP_K = 6
_SCALING = 1.5
_MODES = ("weights", "logits", "combined")
_IMPLEMENTATIONS = ("golden", "mega_gate")


def _golden_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the released V4.1 text Gate with its FP32 projection."""
    logits = functional.linear(hidden.float(), weight.float())  # pylint: disable=not-callable
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores + correction_bias, _TOP_K, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * _SCALING, indices


def _candidate_forward(
    gate: MegaGate,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invoke differentiable HyperMegaGate with model-facing BF16 inputs."""
    return gate(hidden)


def _make_backward_call(
    *,
    mode: str,
    logits: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    logits_gradient: torch.Tensor,
    routing_gradient: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor | None, ...]]:
    """Build a repeatable VJP call while retaining the prebuilt forward graph."""
    inputs = (hidden, weight, correction_bias)
    if mode == "weights":
        outputs = (routing_weights,)
        gradients = (routing_gradient,)
    elif mode == "logits":
        outputs = (logits,)
        gradients = (logits_gradient,)
    elif mode == "combined":
        outputs = (logits, routing_weights)
        gradients = (logits_gradient, routing_gradient)
    else:
        raise ValueError(f"unsupported backward mode: {mode}")

    def backward_call() -> tuple[torch.Tensor | None, ...]:
        """Compute gradients without accumulating into leaf ``.grad`` buffers."""
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=gradients,
            retain_graph=True,
            allow_unused=True,
        )

    return backward_call


def _summarize_kernels(csv_path: Path, iterations: int) -> dict[str, object]:
    """Aggregate raw CANN kernel records into per-backward counts and times."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any("Name" not in row or "Duration(us)" not in row for row in rows):
        raise RuntimeError(f"invalid profiler kernel details: {csv_path}")

    calls = Counter(row["Name"] for row in rows)
    durations: dict[str, float] = {}
    block_nums: dict[str, set[int]] = {}
    for row in rows:
        name = row["Name"]
        durations[name] = durations.get(name, 0.0) + float(row["Duration(us)"])
        block_num = row.get("Block Num", "")
        if block_num:
            block_nums.setdefault(name, set()).add(int(block_num))
    return {
        "device_ms_per_call": sum(durations.values()) / (iterations * 1000.0),
        "kernels_per_call": len(rows) / iterations,
        "kernel_calls_per_call": {name: count / iterations for name, count in calls.items()},
        "kernel_us_per_call": {name: duration / iterations for name, duration in durations.items()},
        "kernel_block_nums": {name: sorted(values) for name, values in block_nums.items()},
    }


def _summarize_operators(csv_path: Path, iterations: int) -> dict[str, object]:
    """Aggregate Host operators recorded while autograd executes the backward."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    calls = Counter(row["Name"] for row in rows)
    host_us: dict[str, float] = {}
    device_us: dict[str, float] = {}
    for row in rows:
        name = row["Name"]
        host_us[name] = host_us.get(name, 0.0) + float(row.get("Host Self Duration(us)", 0.0) or 0.0)
        device_us[name] = device_us.get(name, 0.0) + float(row.get("Device Self Duration(us)", 0.0) or 0.0)
    return {
        "operator_calls_per_call": {name: count / iterations for name, count in calls.items()},
        "operator_host_us_per_call": {name: duration / iterations for name, duration in host_us.items()},
        "operator_device_us_per_call": {name: duration / iterations for name, duration in device_us.items()},
    }


def _profile_backward(
    backward_call: Callable[[], tuple[torch.Tensor | None, ...]],
    *,
    output_dir: Path,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Warm up one retained graph, then export a backward-only CANN profile."""
    for _ in range(warmup):
        backward_call()
    torch.npu.synchronize()

    start_ns = time.perf_counter_ns()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as profiler:
        for _ in range(iterations):
            backward_call()
        torch.npu.synchronize()
    elapsed_ms_per_call = (time.perf_counter_ns() - start_ns) / (iterations * 1.0e6)
    profiler.prof_if.analyse()

    source_dir = Path(profiler.prof_if.prof_path) / "ASCEND_PROFILER_OUTPUT"
    required = ("kernel_details.csv", "operator_details.csv", "trace_view.json")
    if any(not (source_dir / filename).is_file() for filename in required):
        raise RuntimeError(f"incomplete CANN profiler output: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        shutil.copyfile(source_dir / filename, output_dir / filename)

    result = _summarize_kernels(output_dir / "kernel_details.csv", iterations)
    result.update(_summarize_operators(output_dir / "operator_details.csv", iterations))
    result.update(
        elapsed_ms_per_call=elapsed_ms_per_call,
        raw_profile_dir=str(output_dir.resolve()),
        trace_path=str((output_dir / "trace_view.json").resolve()),
    )
    return result


def _run_case(
    *,
    tokens: int,
    mode: str,
    implementation: str,
    device: torch.device,
    output_dir: Path,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Create one golden graph and profile only its selected backward path."""
    hidden = torch.randn(
        (tokens, _HIDDEN), device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    weight = torch.randn(
        (_EXPERTS, _HIDDEN), device=device, dtype=torch.bfloat16, requires_grad=True,
    )
    correction_bias = torch.empty(
        (_EXPERTS,), device=device, dtype=torch.float32,
    ).uniform_(-0.5, 0.5).requires_grad_()
    if implementation == "golden":
        logits, routing_weights, indices = _golden_forward(hidden, weight, correction_bias)
    else:
        gate = MegaGate(
            hidden_size=_HIDDEN,
            num_experts=_EXPERTS,
            top_k=_TOP_K,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=_SCALING,
        )
        gate.weight = torch.nn.Parameter(weight)
        gate.bias = torch.nn.Parameter(correction_bias)
        weight = gate.weight
        correction_bias = gate.bias
        logits, routing_weights, indices = _candidate_forward(gate, hidden)
    logits_gradient = torch.randn_like(logits)
    routing_gradient = torch.randn_like(routing_weights)
    torch.npu.synchronize()

    backward_call = _make_backward_call(
        mode=mode,
        logits=logits,
        routing_weights=routing_weights,
        hidden=hidden,
        weight=weight,
        correction_bias=correction_bias,
        logits_gradient=logits_gradient,
        routing_gradient=routing_gradient,
    )
    gradients = backward_call()
    torch.npu.synchronize()
    hidden_gradient, weight_gradient, correction_bias_gradient = gradients
    if hidden_gradient is None or weight_gradient is None:
        raise RuntimeError(f"{implementation} {mode} backward did not produce hidden and weight gradients")
    if correction_bias_gradient is not None:
        raise RuntimeError("selection-only correction bias unexpectedly received a gradient")

    profile_result = _profile_backward(
        backward_call,
        output_dir=output_dir / f"T{tokens}_{mode}",
        warmup=warmup,
        iterations=iterations,
    )
    return {
        "tokens": tokens,
        "mode": mode,
        "implementation": implementation,
        "outputs": {
            "logits": list(logits.shape),
            "routing_weights": list(routing_weights.shape),
            "expert_indices": list(indices.shape),
        },
        "gradients": {
            "hidden": {"shape": list(hidden_gradient.shape), "dtype": str(hidden_gradient.dtype)},
            "weight": {"shape": list(weight_gradient.shape), "dtype": str(weight_gradient.dtype)},
            "correction_bias": None,
        },
        "profile": profile_result,
    }


def main() -> int:
    """Profile one V4.1 Gate backward implementation for the measured shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--tokens", type=int, choices=_TOKENS, nargs="+")
    parser.add_argument("--mode", choices=_MODES, nargs="+", default=["weights"])
    parser.add_argument("--implementation", choices=_IMPLEMENTATIONS, default="golden")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if not torch.npu.is_available():
        raise RuntimeError("the V4.1 backward probe requires an Ascend NPU")

    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    torch.manual_seed(18)
    torch.npu.manual_seed(18)
    tokens_to_run = tuple(args.tokens) if args.tokens else _TOKENS
    summary_path = args.summary or args.output_dir / f"mega_gate_v41_{args.implementation}_backward.json"
    cases: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "scope": f"{args.implementation} DeepSeek-V4.1 Gate backward only",
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device_index),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cases": cases,
    }
    for tokens in tokens_to_run:
        for mode in args.mode:
            result = _run_case(
                tokens=tokens,
                mode=mode,
                implementation=args.implementation,
                device=device,
                output_dir=args.output_dir / args.implementation,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            cases.append(result)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            profile_result = result["profile"]
            print(
                f"T={tokens} mode={mode}: device={profile_result['device_ms_per_call']:.4f} ms, "
                f"kernels={profile_result['kernels_per_call']:.1f}, "
                f"trace={profile_result['trace_path']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
