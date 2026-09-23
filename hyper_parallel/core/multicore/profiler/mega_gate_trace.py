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
"""Export the actual CANN Device timeline for HyperMegaGate."""

from __future__ import annotations

from collections.abc import Callable
import csv
from pathlib import Path
import shutil
from typing import Any

import torch


def _summarize_device_stages(rows: list[dict[str, str]], iterations: int) -> tuple[dict[str, float], str]:
    """Validate HyperMegaGate's two Casts, Matmul, and Route path."""
    if len(rows) != iterations * 4:
        raise RuntimeError(f"expected four HyperMegaGate Device kernels per call, got {len(rows)} rows")
    stages = {"cast": 0.0, "matmul": 0.0, "route": 0.0}
    counts = {name: 0 for name in stages}
    matmul_implementations = set()
    for row in rows:
        name = row["Name"]
        if "Cast" in name:
            stage = "cast"
        elif "MatMulV2" in name or "MatMulV3" in name:
            stage = "matmul"
            matmul_implementations.add("MatMulV3" if "MatMulV3" in name else "MatMulV2")
        elif "HyperMegaGateRoute" in name:
            stage = "route"
        else:
            raise RuntimeError(f"unexpected HyperMegaGate Device kernel: {name}")
        counts[stage] += 1
        stages[stage] += float(row["Duration(us)"]) / iterations
    if counts != {"cast": iterations * 2, "matmul": iterations, "route": iterations}:
        raise RuntimeError(f"unexpected HyperMegaGate Device stage counts: {counts}")
    if len(matmul_implementations) != 1:
        raise RuntimeError(f"HyperMegaGate changed Matmul implementation inside one profile: {matmul_implementations}")
    return stages, matmul_implementations.pop()


def export_mega_gate_trace(
    call: Callable[[], Any],
    output_path: str | Path,
    *,
    warmup: int = 10,
    iterations: int = 5,
) -> dict[str, Any]:
    """Profile HyperMegaGate calls and export CANN's unmodified Chrome trace.

    Args:
        call: Callable that invokes one HyperMegaGate forward on the current
            NPU stream. Inputs and outputs remain owned by the caller.
        output_path: Path for the raw ``trace_view.json`` content.
        warmup: Calls completed before profiling starts.
        iterations: Calls captured in one profiler window.

    Returns:
        Trace path, selected CANN Matmul implementation, and per-stage Device
        duration in microseconds per call. The stage timeline does not contain
        Matmul per-core cycle records.

    Raises:
        ValueError: Warmup or iteration count is invalid.
        RuntimeError: CANN did not export the expected Cast, Matmul, Route
            stages, or did not produce its raw Chrome trace.
    """
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    # torch_npu is optional for CPU-only imports of the multicore profiler.
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
    from torch_npu.profiler import ProfilerActivity, profile  # pylint: disable=import-outside-toplevel

    if not torch.npu.is_available():
        raise RuntimeError("HyperMegaGate profiling requires an Ascend NPU")

    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as profiler:
        for _ in range(iterations):
            call()
        torch.npu.synchronize()
    profiler.prof_if.analyse()

    source_dir = Path(profiler.prof_if.prof_path) / "ASCEND_PROFILER_OUTPUT"
    source_trace = source_dir / "trace_view.json"
    kernel_details = source_dir / "kernel_details.csv"
    operator_details = source_dir / "operator_details.csv"
    if not source_trace.is_file() or not kernel_details.is_file() or not operator_details.is_file():
        raise RuntimeError(f"CANN profiling output is incomplete: {source_dir}")
    with kernel_details.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stages, matmul_implementation = _summarize_device_stages(rows, iterations)
    with operator_details.open(newline="", encoding="utf-8") as handle:
        operator_rows = list(csv.DictReader(handle))
    gate_calls = sum(row.get("Name") == "aclnnHyperMegaGateRoute" for row in operator_rows)
    if gate_calls != iterations:
        raise RuntimeError(f"expected one aclnnHyperMegaGateRoute call per iteration, got {gate_calls}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_trace.resolve() != target.resolve():
        shutil.copyfile(source_trace, target)
    return {
        "trace_path": str(target),
        "kernel_count": len(rows),
        "matmul_implementation": matmul_implementation,
        "stage_us_per_call": stages,
    }
