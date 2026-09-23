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
"""Single-card precision worker for the formal MegaGate module entry."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import pytest
import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
import torch_npu  # noqa: F401  pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.core.multicore.modules.mega_gate.profiling import MEGA_GATE_ROUTE_GRAD_STAGE_NAMES
from hyper_parallel.core.multicore import profiler as multicore_profiler
from hyper_parallel.core.multicore.torch import mega_gate_ops
# pylint: enable=wrong-import-position

_DEVICE = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
_HIDDEN = 5120
_EXPERTS = 384
_TOP_K = 6
_SCALING = 1.5


def _training_gate(hidden_size: int, experts: int, top_k: int, dtype: torch.dtype) -> MegaGate:
    """Build a small native vision router with FP32 correction biases."""
    gate = MegaGate(
        hidden_size=hidden_size, num_experts=experts, top_k=top_k,
        scoring_func="sqrtsoftplus", routed_scaling_factor=_SCALING, vision_enabled=True,
    ).to(_DEVICE)
    weight = torch.randn((experts, hidden_size), device=_DEVICE, dtype=dtype) * 0.02
    gate.weight = torch.nn.Parameter(weight)
    with torch.no_grad():
        gate.bias.uniform_(-0.1, 0.1)
        gate.bias_vl.uniform_(-0.1, 0.1)
    return gate


def _training_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Weight gradients by expert ID so unsorted TopK order cannot affect the loss."""
    logits, weights, indices = outputs
    coefficients = (indices.float() + 1) / logits.shape[-1]
    return (weights * coefficients).sum() + 0.01 * logits.square().mean()


def _gate_reference(gate: MegaGate, hidden: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    """Use the independent Torch golden with the same parameters and vision mask."""
    if mask is None or gate.bias_vl is None:
        return _golden_config(hidden, gate.weight, gate.bias, gate.top_k, gate.routed_scaling_factor)
    return _golden_vision(hidden, gate.weight, gate.bias, gate.bias_vl, mask, gate.top_k, gate.routed_scaling_factor)


def _check_training_parity(
    gate: MegaGate, hidden: torch.Tensor, mask: torch.Tensor | None,
    *, autocast_enabled: bool = False, mode: str = "combined",
) -> None:
    """Compare independent, synchronized graphs including the FP32 Route boundary."""
    snapshots = []
    for native in (False, True):
        leaf = hidden.detach().clone().requires_grad_()
        with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = gate(leaf, image_mask=mask) if native else _gate_reference(gate, leaf, mask)
            logits, weights, indices = outputs
            loss = (weights * ((indices.float() + 1) / gate.num_experts)).sum()
            if mode == "combined":
                loss = loss + 0.01 * logits.square().mean()
        gradients = torch.autograd.grad(loss, (logits, leaf, gate.weight))
        torch.npu.synchronize()
        snapshots.append((
            tuple(tensor.detach().cpu() for tensor in outputs),
            tuple(tensor.detach().cpu() for tensor in gradients),
        ))
    expected, actual = snapshots
    _assert_outputs_match(actual[0], expected[0])
    assert tuple(t.dtype for t in actual[0]) == (torch.float32, torch.float32, torch.int64), (
        f"output dtypes={tuple(t.dtype for t in actual[0])}, expected FP32/FP32/INT64"
    )
    for index, (gradient, reference) in enumerate(zip(actual[1], expected[1])):
        rtol, atol = (1e-2, 1e-4) if gradient.dtype == torch.bfloat16 else (1e-5, 1e-6)
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol, msg=f"gradient index={index}")
    assert gate.bias.grad is None and (gate.bias_vl is None or gate.bias_vl.grad is None), (
        f"ranking bias gradients must be None: text={gate.bias.grad}, vision={gate.bias_vl}"
    )


def _parameter_test_gate(vision: bool) -> MegaGate:
    """Construct the small parameter-lifecycle test configuration on the current device."""
    return MegaGate(
        hidden_size=128, num_experts=33, top_k=7, scoring_func="sqrtsoftplus",
        routed_scaling_factor=_SCALING, vision_enabled=vision,
    )


@pytest.mark.parametrize("vision", (False, True), ids=("text", "vision"))
def test_mega_gate_state_dict_roundtrip(vision: bool) -> None:
    """Strictly load golden parameters and preserve outputs/gradients across serialization."""
    torch.manual_seed(2310)
    state = {"weight": torch.randn(33, 128) * 0.02, "bias": torch.linspace(-0.1, 0.1, 33)}
    if vision:
        state["bias_vl"] = -state["bias"]
    gate = _parameter_test_gate(vision).to(_DEVICE)
    gate.load_state_dict(state, strict=True)
    hidden = torch.randn((1, 65, 128), device=_DEVICE, dtype=torch.bfloat16)
    mask = torch.arange(65, device=_DEVICE).reshape(1, 65) % 4 == 0 if vision else None
    _check_training_parity(gate, hidden, mask)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "gate.pt"
        torch.save(gate.state_dict(), checkpoint_path)
        restored = _parameter_test_gate(vision).to(_DEVICE)
        restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    assert set(restored.state_dict()) == set(state), (
        f"checkpoint keys={set(restored.state_dict())}, expected={set(state)}"
    )
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter.cpu(), state[name], rtol=0, atol=0)
    _check_training_parity(restored, hidden, mask)


@pytest.mark.parametrize("vision", (False, True), ids=("text", "vision"))
@pytest.mark.parametrize("initialize", ("reset", "load"))
def test_mega_gate_meta_materialization(vision: bool, initialize: str) -> None:
    """Materialize meta parameters using either initialization or checkpoint loading."""
    torch.manual_seed(2311)
    source = _parameter_test_gate(vision)
    with torch.device("meta"):
        gate = _parameter_test_gate(vision)
    assert all(parameter.is_meta for parameter in gate.parameters()), f"expected meta parameters: {gate}"
    gate.to_empty(device=_DEVICE)
    if initialize == "reset":
        gate.reset_parameters()
    else:
        gate.load_state_dict(source.state_dict(), strict=True)
    for name, parameter in gate.named_parameters():
        assert not parameter.is_meta and parameter.device == _DEVICE, (
            f"parameter={name}, device={parameter.device}, expected={_DEVICE}"
        )
        assert bool(torch.isfinite(parameter).all().item()), f"nonfinite parameter={name}, mode={initialize}"
        if initialize == "load":
            torch.testing.assert_close(parameter.cpu(), source.state_dict()[name], rtol=0, atol=0)
        if name != "weight":
            torch.testing.assert_close(parameter, torch.zeros_like(parameter), rtol=0, atol=0)
    assert bool((gate.weight != 0).any().item()), f"weight must be initialized: mode={initialize}"
    hidden = torch.randn((1, 65, 128), device=_DEVICE, dtype=torch.bfloat16)
    mask = torch.arange(65, device=_DEVICE).reshape(1, 65) % 4 == 0 if vision else None
    _check_training_parity(gate, hidden, mask)


@pytest.mark.parametrize("mode", ("weights", "combined"))
@pytest.mark.parametrize("weight_dtype", (torch.float32, torch.bfloat16), ids=("fp32-weight", "bf16-weight"))
@pytest.mark.parametrize("autocast_enabled", (False, True), ids=("no-autocast", "autocast"))
def test_mega_gate_autocast_contract(mode: str, weight_dtype: torch.dtype, autocast_enabled: bool) -> None:
    """Expose dtype contract failures under the caller's unchanged autocast context."""
    torch.manual_seed(2312)
    gate = _training_gate(_HIDDEN, _EXPERTS, _TOP_K, weight_dtype)
    hidden = torch.randn((1, 2048, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    mask = torch.arange(2048, device=_DEVICE).reshape(1, 2048) % 4 == 0
    native_route = mega_gate_ops.mega_gate_vision_route

    def checked_route(logits, text_bias, vision_bias, *args, **kwargs):
        dtypes = (logits.dtype, text_bias.dtype, vision_bias.dtype)
        assert dtypes == (torch.float32,) * 3, (
            f"native FP32 contract failed: logits/text_bias/vision_bias={dtypes}, "
            f"autocast={autocast_enabled}, weight_dtype={weight_dtype}"
        )
        return native_route(logits, text_bias, vision_bias, *args, **kwargs)

    with patch.object(mega_gate_ops, "mega_gate_vision_route", side_effect=checked_route) as launch:
        _check_training_parity(gate, hidden, mask, autocast_enabled=autocast_enabled, mode=mode)
        assert launch.call_count == 1, f"native calls={launch.call_count}, expected=1"


def test_mega_gate_correction_bias_update() -> None:
    """Observe in-place text/vision ranking updates without changing unbiased scores."""
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    hidden = torch.zeros((1, 65, 128), device=_DEVICE, dtype=torch.bfloat16)
    mask = torch.arange(65, device=_DEVICE).reshape(1, 65) % 4 == 0
    base_bias = -torch.arange(33, device=_DEVICE, dtype=torch.float32) / 100
    logits_before = None
    for step in range(3):
        with torch.no_grad():
            if step == 0:
                gate.bias.copy_(base_bias)
                gate.bias_vl.copy_(base_bias)
            elif step == 1:
                gate.bias[7:14].add_(4)
            else:
                gate.bias_vl[14:21].add_(4)
            expected = _gate_reference(gate, hidden, mask)
            torch.npu.synchronize()
            actual = gate(hidden, image_mask=mask)
            torch.npu.synchronize()
            _assert_outputs_match(actual, expected)
            if logits_before is None:
                logits_before = actual[0].clone()
            torch.testing.assert_close(actual[0], logits_before, rtol=0, atol=0)
            text_start = 7 if step >= 1 else 0
            vision_start = 14 if step == 2 else 0
            expected_indices = torch.where(
                mask.reshape(-1, 1),
                torch.arange(vision_start, vision_start + 7, device=_DEVICE),
                torch.arange(text_start, text_start + 7, device=_DEVICE),
            )
            torch.testing.assert_close(actual[2].sort(dim=-1).values, expected_indices, rtol=0, atol=0)
        gate.zero_grad(set_to_none=True)
        _training_loss(gate(hidden, image_mask=mask)).backward()
        torch.npu.synchronize()
        assert gate.bias.grad is None and gate.bias_vl.grad is None, (
            f"ranking bias gradients after update={step}: text={gate.bias.grad}, vision={gate.bias_vl.grad}"
        )


@pytest.mark.parametrize("experts", (31, 32, 33, 63, 64, 65))
@pytest.mark.parametrize("vision", (False, True), ids=("text", "vision"))
def test_mega_gate_topk_padding(experts: int, vision: bool) -> None:
    """Reject padded experts when the first score is largest and DMA rows are unaligned."""
    gate = _training_gate(64, experts, 7, torch.float32)
    hidden = torch.zeros((1, 65, 64), device=_DEVICE)
    mask = torch.ones((1, 65), device=_DEVICE, dtype=torch.bool)
    with torch.no_grad():
        bias = torch.linspace(2.0, 0.0, experts, device=_DEVICE)
        gate.bias.copy_(bias)
        gate.bias_vl.copy_(bias)
        expected = _gate_reference(gate, hidden, mask)
        torch.npu.synchronize()
        actual = gate(hidden, image_mask=mask if vision else None)
        torch.npu.synchronize()
        indices = actual[2].cpu()
        assert bool(((indices >= 0) & (indices < experts)).all()), (
            f"TopK selected padding: experts={experts}, indices={indices.tolist()}"
        )
        _assert_outputs_match(actual, expected)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32), ids=("bf16", "fp32"))
@pytest.mark.parametrize("hidden_size,experts,top_k", ((64, 16, 1), (96, 33, 7), (128, 65, 9), (64, 16, 16)))
def test_mega_gate_dynamic_training_graphs(dtype: torch.dtype, hidden_size: int, experts: int, top_k: int) -> None:
    """Keep multiple layers and shapes alive, then differentiate them in reverse order."""
    torch.manual_seed(2301)
    torch.npu.manual_seed(2301)
    gates = [_training_gate(hidden_size, experts, top_k, dtype) for _ in range(2)]
    pending = []
    # Includes T smaller than the AIV count, tail rows, and multiple leading dimensions.
    for index, leading in enumerate(((1, 1), (1, 17), (1, 41), (1, 65), (2, 3, 11), (1, 17))):
        gate = gates[index % len(gates)]
        hidden = torch.randn((*leading, hidden_size), device=_DEVICE, dtype=dtype, requires_grad=True)
        if len(leading) == 3:
            hidden = hidden.transpose(0, 1).detach().requires_grad_()
        mask = torch.rand(hidden.shape[:-1], device=_DEVICE) < (0.0, 1.0, 0.25)[index % 3]
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference = _gate_reference(gate, reference_hidden, mask)
        reference_grad = torch.autograd.grad(_training_loss(reference), (reference_hidden, gate.weight))
        torch.npu.synchronize()
        expected = tuple(tensor.detach().cpu() for tensor in reference)
        expected_grad = tuple(tensor.detach().cpu() for tensor in reference_grad)
        del reference, reference_grad, reference_hidden
        actual = gate(hidden, image_mask=mask)
        torch.npu.synchronize()
        _assert_outputs_match(tuple(tensor.detach().cpu() for tensor in actual), expected)
        pending.append((gate, hidden, actual, expected, expected_grad))

    for gate, hidden, actual, expected, expected_grad in reversed(pending):
        gradients = torch.autograd.grad(_training_loss(actual), (hidden, gate.weight))
        torch.npu.synchronize()
        _assert_outputs_match(tuple(tensor.detach().cpu() for tensor in actual), expected)
        for gradient, reference in zip(gradients, expected_grad):
            rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-4)
            torch.testing.assert_close(gradient.cpu(), reference, rtol=rtol, atol=atol)


def _async_accumulation_step(
    gate: MegaGate, optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]], native: bool,
) -> torch.Tensor:
    """Enqueue two microbatches and one update; return device-resident finite checks."""
    optimizer.zero_grad(set_to_none=True)
    checks = []
    for hidden, mask in microbatches:
        leaf = hidden.detach().clone().requires_grad_()
        outputs = gate(leaf, image_mask=mask) if native else _gate_reference(gate, leaf, mask)
        loss = _training_loss(outputs) / len(microbatches)
        loss.backward()
        checks.extend((torch.isfinite(loss.detach()), torch.isfinite(leaf.grad).all()))
    checks.append(torch.isfinite(gate.weight.grad).all())
    optimizer.step()
    checks.append(torch.isfinite(gate.weight).all())
    optimizer.zero_grad(set_to_none=True)
    return torch.stack(checks)


def _check_async_training(
    gate: MegaGate, microbatches: list[tuple[torch.Tensor, torch.Tensor]], native: bool,
) -> dict[str, int]:
    """Warm optimizer state, then check a 20-step window without host reads between steps."""
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3)
    _async_accumulation_step(gate, optimizer, microbatches, native)
    torch.npu.synchronize()
    initial_weight = gate.weight.detach().clone()
    gc.collect()
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated(_DEVICE)
    checks = []
    for _ in range(20):
        checks.append(_async_accumulation_step(gate, optimizer, microbatches, native))
    torch.npu.synchronize()
    finite = torch.stack(checks).cpu()
    del checks
    gc.collect()
    torch.npu.synchronize()
    live = torch.npu.memory_allocated(_DEVICE)
    assert bool(finite.all()), f"nonfinite training values: native={native}, failed={torch.nonzero(~finite).tolist()}"
    assert not torch.equal(gate.weight, initial_weight), f"weight did not change after 20 steps: native={native}"
    assert gate.bias.grad is None and gate.bias_vl.grad is None, (
        f"ranking bias gradients: native={native}, text={gate.bias.grad}, vision={gate.bias_vl.grad}"
    )
    assert live <= baseline + 1024 * 1024, (
        f"retained allocations: native={native}, baseline={baseline}, live={live}, tolerance={1024 * 1024}"
    )
    return {"baseline_allocated_bytes": baseline, "live_allocated_bytes": live, "steps": 20}


def test_mega_gate_async_training_steps() -> None:
    """Compare one step, then independently run golden/native AdamW accumulation windows."""
    torch.manual_seed(2313)
    gate = _training_gate(128, 33, 7, torch.bfloat16)
    reference_gate = _training_gate(128, 33, 7, torch.bfloat16)
    reference_gate.load_state_dict(gate.state_dict(), strict=True)
    microbatches = [
        (torch.randn((1, tokens, 128), device=_DEVICE, dtype=torch.bfloat16),
         torch.arange(tokens, device=_DEVICE).reshape(1, tokens) % 4 == 0)
        for tokens in (257, 513)
    ]
    _check_training_parity(gate, *microbatches[0])
    reports = {"golden": _check_async_training(reference_gate, microbatches, False)}
    torch.npu.synchronize()
    reports["candidate"] = _check_async_training(gate, microbatches, True)
    print("MEGA_GATE_ASYNC_TRAINING=" + json.dumps(reports, sort_keys=True), flush=True)


def _gradient_difference(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> dict:
    """Summarize a diagnostic comparison without weakening the test assertion."""
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    mismatch = ~torch.isclose(actual, expected, rtol=rtol, atol=atol)
    return {
        "mismatched_elements": int(mismatch.sum()),
        "max_abs_error": float((actual - expected).abs().max()),
        "nonfinite_actual": int((~torch.isfinite(actual)).sum()),
        "nonfinite_expected": int((~torch.isfinite(expected)).sum()),
    }


def _checkpoint_eager_gradients(gate: MegaGate, hidden: torch.Tensor, mask: torch.Tensor, native: bool) -> tuple:
    """Replay one microbatch and expose gradients before and after the weight Cast."""
    leaf = hidden.detach().requires_grad_()
    projection_weights = []
    linear = functional.linear

    def capture_projection(inputs, weight, bias=None):
        projection_weights.append(weight)
        return linear(inputs, weight, bias)

    with patch.object(functional, "linear", side_effect=capture_projection):
        outputs = gate(leaf, image_mask=mask) if native else _gate_reference(gate, leaf, mask)
    assert len(projection_weights) == 1, "diagnostic expected exactly one projection"
    gradients = torch.autograd.grad(
        _training_loss(outputs) / 2, (outputs[0], projection_weights[0], gate.weight),
    )
    torch.npu.synchronize()
    return tuple(gradient.detach().cpu() for gradient in gradients)


def _report_checkpoint_accumulation(
    gate: MegaGate, reference_gate: MegaGate, microbatches: list,
    *, step: int, use_reentrant: bool,
) -> None:
    """Locate Route, projection, BF16 accumulation or checkpoint divergence on failure only."""
    actual, expected = gate.weight.grad.detach().cpu(), reference_gate.weight.grad.detach().cpu()
    coordinates = (~torch.isclose(actual.float(), expected.float(), rtol=1e-2, atol=1e-4)).nonzero()[:8]
    sample = tuple(coordinates.T)
    report = {
        "step": step, "use_reentrant": use_reentrant, "coordinates": coordinates.tolist(),
        "checkpoint_accumulated": actual[sample].float().tolist(),
        "golden_accumulated": expected[sample].float().tolist(), "microbatches": [],
    }
    sums = [torch.zeros_like(actual), torch.zeros_like(expected)]
    for hidden, mask in microbatches:
        golden = _checkpoint_eager_gradients(reference_gate, hidden, mask, False)
        native = _checkpoint_eager_gradients(gate, hidden, mask, True)
        entry = {}
        for index, name in enumerate(("dlogits", "weight_fp32", "weight_bf16")):
            rtol, atol = (1e-2, 1e-4) if index == 2 else (1e-5, 1e-6)
            entry[name] = _gradient_difference(native[index], golden[index], rtol=rtol, atol=atol)
            if index:
                entry[name].update(native=native[index][sample].float().tolist(),
                                   golden=golden[index][sample].float().tolist())
        sums[0].add_(native[2])
        sums[1].add_(golden[2])
        report["microbatches"].append(entry)
    report["native_eager_sum"] = sums[0][sample].float().tolist()
    report["golden_eager_sum"] = sums[1][sample].float().tolist()
    report["checkpoint_vs_native_eager"] = _gradient_difference(actual, sums[0], rtol=1e-2, atol=1e-4)
    report["golden_vs_replay"] = _gradient_difference(expected, sums[1], rtol=1e-2, atol=1e-4)
    print("MEGA_GATE_CHECKPOINT_DIAGNOSTIC=" + json.dumps(report, sort_keys=True), flush=True)


@pytest.mark.parametrize("use_reentrant", (False, True), ids=("nonreentrant", "reentrant"))
@pytest.mark.parametrize("model_shape", (False, True), ids=("small", "model-4k"))
def test_mega_gate_checkpoint_accumulation(use_reentrant: bool, model_shape: bool) -> None:
    """Compare checkpoint recompute, microbatch accumulation, and optimizer updates with golden."""
    torch.manual_seed(2302)
    torch.npu.manual_seed(2302)
    hidden_size, experts, top_k = (_HIDDEN, _EXPERTS, _TOP_K) if model_shape else (128, 33, 7)
    token_counts = (4096, 4096) if model_shape else (65, 129)
    gate = _training_gate(hidden_size, experts, top_k, torch.bfloat16)
    reference_gate = _training_gate(hidden_size, experts, top_k, torch.bfloat16)
    reference_gate.load_state_dict(gate.state_dict())
    optimizer = torch.optim.SGD(gate.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.SGD(reference_gate.parameters(), lr=1e-3)
    for step in range(2 if model_shape else 3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for tokens in token_counts:
            hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16, requires_grad=True)
            mask = torch.rand((1, tokens), device=_DEVICE) < 0.25
            microbatches.append((hidden.detach(), mask))
            reference_hidden = hidden.detach().clone().requires_grad_()
            expected = _gate_reference(reference_gate, reference_hidden, mask)
            (_training_loss(expected) / 2).backward()
            torch.npu.synchronize()
            with patch.object(
                mega_gate_ops, "mega_gate_vision_route", wraps=mega_gate_ops.mega_gate_vision_route,
            ) as launch:
                actual = checkpoint(gate, hidden, mask, use_reentrant=use_reentrant)
                torch.npu.synchronize()
                _assert_outputs_match(actual, expected)
                (_training_loss(actual) / 2).backward()
                torch.npu.synchronize()
                assert launch.call_count == 2, (
                    f"expected native forward plus recompute: calls={launch.call_count}, reentrant={use_reentrant}"
                )
            _assert_outputs_match(actual, expected)
            torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-2, atol=1e-4)
        try:
            torch.testing.assert_close(gate.weight.grad, reference_gate.weight.grad, rtol=1e-2, atol=1e-4)
        except AssertionError:
            _report_checkpoint_accumulation(
                gate, reference_gate, microbatches, step=step, use_reentrant=use_reentrant,
            )
            raise
        assert gate.bias.grad is None and gate.bias_vl.grad is None, (
            f"ranking biases must have no gradients: text={gate.bias.grad}, vision={gate.bias_vl.grad}"
        )
        optimizer.step()
        reference_optimizer.step()
        torch.npu.synchronize()
        torch.testing.assert_close(gate.weight, reference_gate.weight, rtol=1e-2, atol=1e-4)


def _memory_training_step(gate: MegaGate, tokens: int, native: bool) -> None:
    """Release each fresh graph before measuring live allocations in the caller."""
    hidden = torch.randn((1, tokens, gate.hidden_size), device=_DEVICE, dtype=gate.weight.dtype, requires_grad=True)
    mask = torch.rand((1, tokens), device=_DEVICE) < 0.25
    outputs = gate(hidden, image_mask=mask) if native else _gate_reference(gate, hidden, mask)
    _training_loss(outputs).backward()
    gate.zero_grad(set_to_none=True)


@pytest.mark.parametrize("token_counts", ((2048, 8192), (8192, 32768)), ids=("2k-8k", "8k-32k"))
def test_mega_gate_training_memory_stability(token_counts: tuple[int, int]) -> None:
    """Record golden/native peaks and reject retained NPU allocations across fresh training graphs."""
    torch.manual_seed(2303)
    torch.npu.manual_seed(2303)
    gate = _training_gate(_HIDDEN, _EXPERTS, _TOP_K, torch.bfloat16)
    reports = {}
    for native in (False, True):
        # Warm every shape and the shared Plan before comparing live allocations.
        for _ in range(3):
            for tokens in token_counts:
                _memory_training_step(gate, tokens, native)
                torch.npu.synchronize()
        gc.collect()
        torch.npu.synchronize()
        baseline = torch.npu.memory_allocated(_DEVICE)
        torch.npu.reset_peak_memory_stats(_DEVICE)
        live_bytes = []
        for _ in range(5):
            for tokens in token_counts:
                _memory_training_step(gate, tokens, native)
                torch.npu.synchronize()
            gc.collect()
            torch.npu.synchronize()
            live_bytes.append(torch.npu.memory_allocated(_DEVICE))
        peak = torch.npu.max_memory_allocated(_DEVICE)
        reports["candidate" if native else "golden"] = {
            "token_counts": token_counts,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak,
            "peak_increment_bytes": peak - baseline,
            "live_allocated_bytes_after_cycles": live_bytes,
            "reserved_bytes": torch.npu.memory_reserved(_DEVICE),
        }
        # Reserved allocator cache is reusable; only live allocations indicate retained tensors.
        tolerance = 1024 * 1024
        assert max(live_bytes) <= baseline + tolerance, (
            f"training allocations did not return to baseline: native={native}, baseline={baseline}, "
            f"live={live_bytes}, tolerance={tolerance}"
        )
    print("MEGA_GATE_TRAINING_MEMORY=" + json.dumps(reports, sort_keys=True), flush=True)


def _make_gate(*, scoring: str = "sqrtsoftplus", vision_enabled: bool = False) -> MegaGate:
    """Create the formal V4.1 module configuration."""
    return MegaGate(
        hidden_size=_HIDDEN,
        num_experts=_EXPERTS,
        scoring_func=scoring,
        top_k=_TOP_K,
        routed_scaling_factor=_SCALING,
        vision_enabled=vision_enabled,
    )


def _set_parameters(gate: MegaGate, weight: torch.Tensor, bias: torch.Tensor) -> None:
    """Install deterministic leaves while preserving the module parameter names."""
    gate.weight = torch.nn.Parameter(weight, requires_grad=weight.requires_grad)
    gate.bias = torch.nn.Parameter(bias, requires_grad=bias.requires_grad)


def _golden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the V4.1 Projection and text Route reference."""
    logits = functional.linear(hidden.reshape(-1, _HIDDEN).float(), weight.float())  # pylint: disable=not-callable
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores + bias, _TOP_K, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * _SCALING, indices


def _golden_config(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the reference for an arbitrary compatible Gate shape."""
    logits = functional.linear(  # pylint: disable=not-callable
        hidden.reshape(-1, hidden.shape[-1]).float(), weight.float()
    )
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    if top_k > 1:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * scaling, indices


def _golden_vision(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    text_bias: torch.Tensor,
    vision_bias: torch.Tensor,
    image_mask: torch.Tensor,
    top_k: int = _TOP_K,
    scaling: float = _SCALING,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the V4.1 reference with per-token text or vision bias."""
    logits = functional.linear(  # pylint: disable=not-callable
        hidden.reshape(-1, hidden.shape[-1]).float(), weight.float()
    )
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    correction_bias = torch.where(
        image_mask.reshape(-1, 1),
        vision_bias.unsqueeze(0),
        text_bias.unsqueeze(0),
    )
    indices = torch.topk(scores + correction_bias, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    if top_k > 1:
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, selected * scaling, indices


def _assert_outputs_match(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Compare expert sets and their weights independently of TopK order."""
    actual_logits, actual_weights, actual_indices = actual
    expected_logits, expected_weights, expected_indices = expected
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1.0e-5, atol=1.0e-6)
    actual_order = actual_indices.argsort(dim=-1)
    expected_order = expected_indices.argsort(dim=-1)
    sorted_actual_indices = actual_indices.gather(1, actual_order)
    sorted_expected_indices = expected_indices.gather(1, expected_order)
    matching_indices = sorted_actual_indices == sorted_expected_indices
    if not bool(matching_indices.all().item()):
        mismatched_rows = (~matching_indices).any(dim=1).nonzero().flatten()
        sample_rows = mismatched_rows[:16]
        raise AssertionError(
            "MegaGate expert sets differ: "
            f"mismatched_rows={mismatched_rows.numel()}, "
            f"sample_rows={sample_rows.cpu().tolist()}, "
            f"actual={sorted_actual_indices[sample_rows].cpu().tolist()}, "
            f"expected={sorted_expected_indices[sample_rows].cpu().tolist()}"
        )
    sorted_actual_weights = actual_weights.gather(1, actual_order)
    sorted_expected_weights = expected_weights.gather(1, expected_order)
    close = torch.isclose(sorted_actual_weights, sorted_expected_weights, rtol=1.0e-5, atol=1.0e-6)
    if not bool(close.all().item()):
        mismatched_rows = (~close).any(dim=1).nonzero().flatten()
        sample_rows = mismatched_rows[:16]
        raise AssertionError(
            "MegaGate routing weights differ: "
            f"mismatched_rows={mismatched_rows.numel()}, "
            f"sample_rows={sample_rows.cpu().tolist()}, "
            f"actual={sorted_actual_weights[sample_rows].cpu().tolist()}, "
            f"expected={sorted_expected_weights[sample_rows].cpu().tolist()}, "
            f"actual_sums={actual_weights[sample_rows].sum(dim=1).cpu().tolist()}, "
            f"expected_sums={expected_weights[sample_rows].sum(dim=1).cpu().tolist()}"
        )


def _make_inputs(batch: int, sequence: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic contiguous model inputs for one supported shape."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    hidden = torch.randn((batch, sequence, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((_EXPERTS, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
    bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    return hidden, weight, bias


def test_mega_gate_module_parity() -> None:
    """Check module flattening and numerical parity on every supported length."""
    gate = _make_gate()
    for seed, shape in enumerate(((1, 2048), (2, 2048), (4, 2048)), start=20):
        hidden, weight, bias = _make_inputs(*shape, seed)
        _set_parameters(gate, weight, bias)
        with torch.no_grad():
            expected = _golden(hidden, weight, bias)
            actual = gate(hidden)
        _assert_outputs_match(actual, expected)


@pytest.mark.parametrize(
    "mask_kind,batch",
    (("text", 1), ("vision", 1), ("mixed", 1), ("mixed", 2), ("mixed", 4)),
    ids=("all-text", "all-vision", "mixed-2k", "mixed-4k", "mixed-8k"),
)
def test_mega_gate_vision_module_parity(mask_kind: str, batch: int) -> None:
    """Compare fused visual bias selection with the V4.1 Torch route."""
    gate = _make_gate(vision_enabled=True)
    hidden, weight, text_bias = _make_inputs(batch, 2048, seed=27 + batch)
    vision_bias = torch.linspace(2.0, -2.0, _EXPERTS, device=_DEVICE, dtype=torch.float32)
    _set_parameters(gate, weight, text_bias)
    gate.bias_vl = torch.nn.Parameter(vision_bias)
    if mask_kind == "text":
        image_mask = torch.zeros(hidden.shape[:-1], device=_DEVICE, dtype=torch.bool)
    elif mask_kind == "vision":
        image_mask = torch.ones(hidden.shape[:-1], device=_DEVICE, dtype=torch.bool)
    else:
        image_mask = torch.arange(hidden.numel() // _HIDDEN, device=_DEVICE).reshape(hidden.shape[:-1]) % 2 == 0
    with torch.no_grad():
        expected = _golden_vision(hidden, weight, text_bias, vision_bias, image_mask)
        actual = gate(hidden, image_mask=image_mask)
    _assert_outputs_match(actual, expected)


def test_mega_gate_ignores_image_mask_without_vision_bias() -> None:
    """Use the text native path when the module has no visual bias parameter."""
    gate = _make_gate(vision_enabled=False)
    hidden, weight, bias = _make_inputs(1, 2048, seed=28)
    _set_parameters(gate, weight, bias)
    image_mask = torch.ones(hidden.shape[:-1], device=_DEVICE, dtype=torch.bool)
    with torch.no_grad():
        expected = _golden(hidden, weight, bias)
        actual = gate(hidden, image_mask=image_mask)
    _assert_outputs_match(actual, expected)


def test_mega_gate_dynamic_vision_configuration_parity() -> None:
    """Cover visual routing with runtime expert, hidden, TopK, and tail dimensions."""
    tokens, hidden_size, expert_count, top_k = 73, 96, 24, 5
    torch.manual_seed(29)
    torch.npu.manual_seed(29)
    hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((expert_count, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    text_bias = torch.linspace(-0.5, 0.5, expert_count, device=_DEVICE)
    vision_bias = torch.linspace(1.0, -1.0, expert_count, device=_DEVICE)
    image_mask = torch.arange(tokens, device=_DEVICE).reshape(1, tokens) % 3 == 1
    gate = MegaGate(
        hidden_size=hidden_size,
        num_experts=expert_count,
        top_k=top_k,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=_SCALING,
        vision_enabled=True,
    )
    _set_parameters(gate, weight, text_bias)
    gate.bias_vl = torch.nn.Parameter(vision_bias)
    with torch.no_grad():
        expected = _golden_vision(hidden, weight, text_bias, vision_bias, image_mask, top_k, _SCALING)
        actual = gate(hidden, image_mask=image_mask)
    _assert_outputs_match(actual, expected)


def test_mega_gate_module_repeat_determinism() -> None:
    """Require stable outputs across repeated calls through the module entry."""
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(2, 2048, seed=31)
    _set_parameters(gate, weight, bias)
    with torch.no_grad():
        first = gate(hidden)
        for _ in range(5):
            repeated = gate(hidden)
            for first_tensor, repeated_tensor in zip(first, repeated):
                assert torch.equal(first_tensor, repeated_tensor), (
                    f"repeated MegaGate output changed: first={first_tensor}, repeated={repeated_tensor}"
                )


def test_mega_gate_pipeline_profile() -> None:
    """Require one invocation to expose ten ordered stages on every active AIV."""
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(1, 2048, seed=36)
    _set_parameters(gate, weight, bias)
    with torch.no_grad():
        for _ in range(5):
            gate(hidden)
        torch.npu.synchronize()
        with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
            gate(hidden)
            profiler.step()
    with tempfile.TemporaryDirectory() as output_dir:
        trace = profiler.export_chrome_trace(Path(output_dir) / "mega_gate_trace.json")
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert metadata["invocationCount"] == 1
    worker_ids = sorted({event["args"]["block_id"] for event in events})
    assert 0 < len(worker_ids) <= 48
    assert worker_ids == list(range(len(worker_ids)))
    assert metadata["recordCount"] == len(worker_ids) * 10
    assert metadata["droppedRecordCount"] == 0
    assert metadata["warnings"] == []
    assert len(events) == len(worker_ids) * 10
    assert len({event["tid"] for event in events}) == len(worker_ids)
    for worker in worker_ids:
        worker_events = [event for event in events if event["args"]["block_id"] == worker]
        assert [event["args"]["task_id"] for event in worker_events] == list(range(10))


def test_mega_gate_grad_pipeline_profile() -> None:
    """Require RouteGrad to expose its ordered token-row stages."""
    gate = _make_gate()
    hidden, weight, bias = _gradient_inputs(*_make_inputs(1, 2048, seed=37))
    _set_parameters(gate, weight, bias)
    logits, routing_weights, _ = gate(hidden)
    logits_gradient = torch.randn_like(logits)
    routing_gradient = torch.randn_like(routing_weights)
    torch.npu.synchronize()
    with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
        torch.autograd.grad(
            (logits, routing_weights),
            (hidden, gate.weight),
            grad_outputs=(logits_gradient, routing_gradient),
        )
        profiler.step()
    with tempfile.TemporaryDirectory() as output_dir:
        trace = profiler.export_chrome_trace(Path(output_dir) / "mega_gate_grad_trace.json")
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    worker_ids = sorted({event["args"]["block_id"] for event in events})
    stage_count = len(MEGA_GATE_ROUTE_GRAD_STAGE_NAMES)
    assert metadata["invocationCount"] == 1
    assert 0 < len(worker_ids) <= 48
    assert metadata["recordCount"] == len(worker_ids) * stage_count
    assert metadata["droppedRecordCount"] == 0
    assert metadata["warnings"] == []
    for worker in worker_ids:
        worker_events = [event for event in events if event["args"]["block_id"] == worker]
        assert [event["args"]["task_id"] for event in worker_events] == list(range(stage_count))


def test_mega_gate_module_support_contract() -> None:
    """Check V4.1 parameter names and config construction."""
    gate = MegaGate.from_config(SimpleNamespace(
        hidden_size=_HIDDEN,
        num_local_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=_SCALING,
        v41_vision_enabled=True,
        initializer_range=0.02,
    ))
    assert tuple(gate.state_dict()) == ("weight", "bias", "bias_vl")
    assert tuple(gate.weight.shape) == (_EXPERTS, _HIDDEN)
    assert tuple(gate.bias.shape) == (_EXPERTS,)
    assert tuple(gate.bias_vl.shape) == (_EXPERTS,)


def test_mega_gate_dynamic_configuration_parity() -> None:
    """Verify dimensions and TopK are runtime configuration rather than a shape whitelist."""
    for seed, (tokens, hidden_size, expert_count, top_k) in enumerate(
        ((65, 64, 16, 1), (73, 96, 24, 5)), start=50
    ):
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
        weight = torch.randn((expert_count, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
        bias = torch.empty((expert_count,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
        gate = MegaGate(
            hidden_size=hidden_size,
            num_experts=expert_count,
            scoring_func="sqrtsoftplus",
            top_k=top_k,
            routed_scaling_factor=_SCALING,
        )
        _set_parameters(gate, weight, bias)
        with torch.no_grad():
            expected = _golden_config(hidden, weight, bias, top_k, _SCALING)
            actual = gate(hidden)
        _assert_outputs_match(actual, expected)


def _gradient_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached leaves so golden and MegaGate build independent graphs."""
    return (
        hidden.detach().clone().requires_grad_(),
        weight.detach().clone().requires_grad_(),
        bias.detach().clone().requires_grad_(),
    )


def _compute_gradients(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mode: str,
    logits_gradient: torch.Tensor,
    routing_gradient: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """Compute one of the three supported Gate vector-Jacobian products."""
    logits, routing_weights, _ = outputs
    if mode == "logits":
        selected_outputs = (logits,)
        selected_gradients = (logits_gradient,)
    elif mode == "weights":
        selected_outputs = (routing_weights,)
        selected_gradients = (routing_gradient,)
    elif mode == "combined":
        selected_outputs = (logits, routing_weights)
        selected_gradients = (logits_gradient, routing_gradient)
    else:
        raise ValueError(f"unsupported gradient mode: {mode}")
    return torch.autograd.grad(
        selected_outputs,
        inputs,
        grad_outputs=selected_gradients,
        allow_unused=True,
    )


def _compute_logits_gradient(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mode: str,
    logits_gradient: torch.Tensor,
    routing_gradient: torch.Tensor,
) -> torch.Tensor:
    """Return golden's FP32 gradient at the Projection/Route boundary."""
    logits, routing_weights, _ = outputs
    if mode == "logits":
        selected_outputs = (logits,)
        selected_gradients = (logits_gradient,)
    elif mode == "weights":
        selected_outputs = (routing_weights,)
        selected_gradients = (routing_gradient,)
    elif mode == "combined":
        selected_outputs = (logits, routing_weights)
        selected_gradients = (logits_gradient, routing_gradient)
    else:
        raise ValueError(f"unsupported gradient mode: {mode}")
    return torch.autograd.grad(
        selected_outputs,
        logits,
        grad_outputs=selected_gradients,
        retain_graph=True,
    )[0]


@pytest.mark.parametrize("mode", ("logits", "weights", "combined"), ids=lambda mode: mode)
@pytest.mark.parametrize(
    "batch,sequence",
    ((1, 2048), (2, 2048), (4, 2048)),
    ids=("2k", "4k", "8k"),
)
def test_mega_gate_backward_parity(mode: str, batch: int, sequence: int) -> None:
    """Compare one synchronized RouteGrad and projection-gradient case."""
    gate = _make_gate()
    hidden, weight, bias = _make_inputs(batch, sequence, seed=60 + batch)
    # Match model initialization scale so the main parity matrix compares
    # finite projection gradients. Dedicated extreme tests cover FP32
    # Softplus underflow and its NaN/Inf contract.
    weight.mul_(0.02)
    mode_golden_inputs = _gradient_inputs(hidden, weight, bias)
    mode_actual_inputs = _gradient_inputs(hidden, weight, bias)
    _set_parameters(gate, mode_actual_inputs[1], mode_actual_inputs[2])
    mode_actual_inputs = (mode_actual_inputs[0], gate.weight, gate.bias)
    mode_golden_outputs = _golden(*mode_golden_inputs)
    mode_actual_outputs = gate(mode_actual_inputs[0])
    _assert_outputs_match(mode_actual_outputs, mode_golden_outputs)
    sorted_indices = mode_actual_outputs[2].sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item())

    logits_gradient = torch.randn_like(mode_golden_outputs[0])
    routing_gradient = torch.randn_like(mode_golden_outputs[1])
    golden_boundary_grad = _compute_logits_gradient(
        mode_golden_outputs,
        mode,
        logits_gradient,
        routing_gradient,
    )
    if mode == "logits":
        actual_boundary_grad = logits_gradient
    else:
        with torch.no_grad():
            plan = gate._plan(mode_actual_inputs[0])  # pylint: disable=protected-access
            native_logits = functional.linear(  # pylint: disable=not-callable
                mode_actual_inputs[0].reshape(-1, _HIDDEN).float(),
                gate.weight.float(),
            )
            native_outputs = mega_gate_ops.mega_gate_route(
                native_logits,
                gate.bias,
                plan.vision_mask_placeholder,
                plan.runtime.normal_tensor,
                plan.runtime.normal_tensor,
                top_k=_TOP_K,
                routed_scaling_factor=_SCALING,
            )
            route_grad = plan.grad_route_runtime
            actual_boundary_grad = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
                native_logits,
                native_outputs[2],
                native_outputs[3],
                native_outputs[4],
                native_outputs[1],
                routing_gradient,
                native_logits,
                route_grad.normal_tensor,
                route_grad.normal_tensor,
                top_k=_TOP_K,
                routed_scaling_factor=_SCALING,
                has_direct_grad=False,
            )
            if mode == "combined":
                actual_boundary_grad = actual_boundary_grad + logits_gradient
        torch.npu.synchronize()
    torch.testing.assert_close(actual_boundary_grad, golden_boundary_grad, rtol=1.0e-5, atol=1.0e-6)

    expected = _compute_gradients(
        mode_golden_outputs,
        mode_golden_inputs,
        mode,
        logits_gradient,
        routing_gradient,
    )
    actual = _compute_gradients(
        mode_actual_outputs,
        mode_actual_inputs,
        mode,
        logits_gradient,
        routing_gradient,
    )
    torch.npu.synchronize()
    assert expected[2] is None and actual[2] is None
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=1.0e-2, atol=1.0e-4)
    torch.testing.assert_close(actual[1].float(), expected[1].float(), rtol=1.0e-2, atol=1.0e-4)


@pytest.mark.parametrize(
    "mode,batch,sequence",
    (
        ("weights", 1, 2048),
        ("weights", 4, 2048),
        ("combined", 1, 2048),
        ("combined", 4, 2048),
    ),
    ids=("weights-mixed-2k", "weights-mixed-8k", "combined-mixed-2k", "combined-mixed-8k"),
)
def test_mega_gate_vision_backward_parity(mode: str, batch: int, sequence: int) -> None:
    """Cover mixed-token vision RouteGrad at the FP32 boundary and BF16 inputs."""
    gate = _make_gate(vision_enabled=True)
    hidden, weight, text_bias = _make_inputs(batch, sequence, seed=69 + batch)
    weight.mul_(0.02)
    vision_bias = torch.linspace(1.5, -1.5, _EXPERTS, device=_DEVICE, dtype=torch.float32)
    token_ids = torch.arange(batch * sequence, device=_DEVICE).reshape(batch, sequence)
    image_mask = (token_ids * 17 + token_ids // 31) % 5 < 2

    golden_hidden = hidden.detach().clone().requires_grad_()
    golden_weight = weight.detach().clone().requires_grad_()
    actual_hidden = hidden.detach().clone().requires_grad_()
    actual_weight = weight.detach().clone().requires_grad_()
    _set_parameters(gate, actual_weight, text_bias.detach().clone())
    gate.bias_vl = torch.nn.Parameter(vision_bias.detach().clone())
    expected = _golden_vision(
        golden_hidden,
        golden_weight,
        text_bias,
        vision_bias,
        image_mask,
    )
    actual = gate(actual_hidden, image_mask=image_mask)
    _assert_outputs_match(actual, expected)
    sorted_indices = actual[2].sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item())

    logits_gradient = torch.randn_like(expected[0])
    routing_gradient = torch.randn_like(expected[1])
    expected_boundary_grad = _compute_logits_gradient(
        expected,
        mode,
        logits_gradient,
        routing_gradient,
    )
    actual_boundary_grad = _compute_logits_gradient(
        actual,
        mode,
        logits_gradient,
        routing_gradient,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_boundary_grad, expected_boundary_grad, rtol=1.0e-5, atol=1.0e-6)

    expected_outputs = (expected[1],) if mode == "weights" else expected[:2]
    actual_outputs = (actual[1],) if mode == "weights" else actual[:2]
    gradients = (routing_gradient,) if mode == "weights" else (logits_gradient, routing_gradient)
    expected_gradients = torch.autograd.grad(
        expected_outputs,
        (golden_hidden, golden_weight),
        grad_outputs=gradients,
    )
    actual_gradients = torch.autograd.grad(
        actual_outputs,
        (actual_hidden, gate.weight),
        grad_outputs=gradients,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_gradients[0].float(), expected_gradients[0].float(), rtol=1.0e-2, atol=1.0e-4)
    torch.testing.assert_close(actual_gradients[1].float(), expected_gradients[1].float(), rtol=1.0e-2, atol=1.0e-4)


def test_mega_gate_vision_weights_only_fresh_graph_regression() -> None:
    """Check the former 8K failure with independently synchronized backends."""
    torch.manual_seed(18)
    torch.npu.manual_seed(18)
    case_order = (
        (2048, "weights"),
        (2048, "combined"),
        (4096, "weights"),
        (4096, "combined"),
        (8192, "weights"),
    )
    for tokens, _ in case_order:
        hidden = torch.randn((1, tokens, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
        weight = torch.randn((_EXPERTS, _HIDDEN), device=_DEVICE, dtype=torch.bfloat16)
        weight.mul_(0.02)
        text_bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
        vision_bias = torch.empty((_EXPERTS,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
        image_mask = (torch.arange(tokens, device=_DEVICE) < round(tokens * 0.25)).reshape(1, tokens)
        gate = _make_gate(vision_enabled=True)
        _ = torch.randn((tokens, _EXPERTS), device=_DEVICE, dtype=torch.float32)
        routing_gradient = torch.randn((tokens, _TOP_K), device=_DEVICE, dtype=torch.float32)

    golden_hidden = hidden.detach().clone().requires_grad_()
    golden_weight = weight.detach().clone().requires_grad_()
    golden_outputs = _golden_vision(
        golden_hidden,
        golden_weight,
        text_bias,
        vision_bias,
        image_mask,
    )
    expected_gradients = torch.autograd.grad(
        (golden_outputs[1],),
        (golden_hidden, golden_weight),
        grad_outputs=(routing_gradient,),
    )
    torch.npu.synchronize()

    actual_hidden = hidden.detach().clone().requires_grad_()
    actual_weight = weight.detach().clone().requires_grad_()
    _set_parameters(gate, actual_weight, text_bias)
    gate.bias_vl = torch.nn.Parameter(vision_bias, requires_grad=False)
    actual_outputs = gate(actual_hidden, image_mask=image_mask)
    actual_gradients = torch.autograd.grad(
        (actual_outputs[1],),
        (actual_hidden, gate.weight),
        grad_outputs=(routing_gradient,),
    )
    torch.npu.synchronize()

    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient.float(),
            rtol=1.0e-2,
            atol=1.0e-4,
        )

    burst_hidden = hidden.detach().clone().requires_grad_()
    burst_weight = weight.detach().clone().requires_grad_()
    _set_parameters(gate, burst_weight, text_bias)
    burst_outputs = gate(burst_hidden, image_mask=image_mask)
    burst_gradients = [
        torch.autograd.grad(
            (burst_outputs[1],),
            (burst_hidden, gate.weight),
            grad_outputs=(routing_gradient,),
            retain_graph=True,
        )
        for _ in range(3)
    ]
    torch.npu.synchronize()
    for gradients in burst_gradients:
        for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
            torch.testing.assert_close(
                actual_gradient.float(),
                expected_gradient.float(),
                rtol=1.0e-2,
                atol=1.0e-4,
            )

def test_mega_gate_dynamic_backward_parity() -> None:
    """Cover K=1, a tail row, dynamic dimensions, and retained backward state."""
    tokens, hidden_size, expert_count, top_k = 65, 64, 16, 1
    torch.manual_seed(72)
    torch.npu.manual_seed(72)
    hidden = torch.randn((1, tokens, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    weight = torch.randn((expert_count, hidden_size), device=_DEVICE, dtype=torch.bfloat16)
    bias = torch.empty((expert_count,), device=_DEVICE, dtype=torch.float32).uniform_(-0.5, 0.5)
    gate = MegaGate(
        hidden_size=hidden_size,
        num_experts=expert_count,
        scoring_func="sqrtsoftplus",
        top_k=top_k,
        routed_scaling_factor=_SCALING,
    )
    golden_inputs = _gradient_inputs(hidden, weight, bias)
    actual_inputs = _gradient_inputs(hidden, weight, bias)
    _set_parameters(gate, actual_inputs[1], actual_inputs[2])
    actual_inputs = (actual_inputs[0], gate.weight, gate.bias)
    golden_outputs = _golden_config(*golden_inputs, top_k, _SCALING)
    actual_outputs = gate(actual_inputs[0])
    logits_gradient = torch.randn_like(golden_outputs[0])
    routing_gradient = torch.randn_like(golden_outputs[1])
    expected = torch.autograd.grad(
        golden_outputs[:2],
        golden_inputs[:2],
        grad_outputs=(logits_gradient, routing_gradient),
        retain_graph=True,
    )
    for _ in range(5):
        actual = torch.autograd.grad(
            actual_outputs[:2],
            actual_inputs[:2],
            grad_outputs=(logits_gradient, routing_gradient),
            retain_graph=True,
        )
        torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=1.0e-2, atol=1.0e-4)
        torch.testing.assert_close(actual[1].float(), expected[1].float(), rtol=1.0e-2, atol=1.0e-4)

    for need_hidden, need_weight in ((True, False), (False, True)):
        branch_hidden = hidden.detach().clone().requires_grad_(need_hidden)
        branch_weight = weight.detach().clone().requires_grad_(need_weight)
        golden_hidden = hidden.detach().clone().requires_grad_(need_hidden)
        golden_weight = weight.detach().clone().requires_grad_(need_weight)
        _set_parameters(gate, branch_weight, bias)
        actual_branch_outputs = gate(branch_hidden)
        golden_branch_outputs = _golden_config(golden_hidden, golden_weight, bias, top_k, _SCALING)
        actual_inputs = tuple(
            tensor for tensor, needed in ((branch_hidden, need_hidden), (gate.weight, need_weight)) if needed
        )
        golden_branch_inputs = tuple(
            tensor for tensor, needed in ((golden_hidden, need_hidden), (golden_weight, need_weight)) if needed
        )
        actual_branch_grad = torch.autograd.grad(
            actual_branch_outputs[:2],
            actual_inputs,
            grad_outputs=(logits_gradient, routing_gradient),
        )
        golden_branch_grad = torch.autograd.grad(
            golden_branch_outputs[:2],
            golden_branch_inputs,
            grad_outputs=(logits_gradient, routing_gradient),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual_branch_grad[0].float(), golden_branch_grad[0].float(), rtol=1.0e-2, atol=1.0e-4
        )


@pytest.mark.parametrize(
    "tokens,top_k",
    ((2048, 6), (8193, 1), (8193, 9), (16384, 7), (32768, 6), (32769, 64)),
    ids=("2k-k6", "8k-tail-k1", "8k-tail-k9", "16k-k7", "32k-k6", "32k-tail-k64"),
)
def test_mega_gate_route_grad_batch_parity(tokens: int, top_k: int) -> None:
    """Check different elementwise/reduction batches without projection MatMuls."""
    torch.manual_seed(2309 + top_k)
    logits = (torch.randn((tokens, _EXPERTS), device=_DEVICE) * 3).requires_grad_()
    scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    indices = torch.topk(scores, top_k, dim=-1, sorted=False).indices
    selected = scores.gather(1, indices)
    denominator = selected.sum(dim=-1, keepdim=True) + 1.0e-20
    weights = selected / denominator if top_k > 1 else selected
    weights = weights * _SCALING
    gradient = torch.randn_like(weights)
    expected = torch.autograd.grad(weights, logits, grad_outputs=gradient)[0]
    # Complete the reference graph before the native call, including on the
    # async task queue used in training.
    torch.npu.synchronize()

    gate = _make_gate()
    plan = gate._plan(logits)  # pylint: disable=protected-access
    runtime = plan.grad_route_k1_runtime if top_k == 1 else plan.grad_route_runtime
    with torch.no_grad():
        actual = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
            logits.detach(), scores.detach(), selected.detach(), denominator.detach(),
            indices, gradient, logits.detach(), runtime.normal_tensor, runtime.normal_tensor,
            top_k=top_k, routed_scaling_factor=_SCALING, has_direct_grad=False,
        )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_mega_gate_route_grad_extremes() -> None:
    """Check stable SoftplusGrad and Scatter semantics at threshold and underflow inputs."""
    top_k = 3
    logits_values = (
        (-120.0, -20.0, -1.0, 0.0, 19.999, 20.0, 20.001, 80.0),
        (-100.0, -2.0, -0.0, 0.5, 19.5, 20.0, 21.0, 60.0),
    )
    reference_logits = torch.tensor(logits_values, device=_DEVICE, dtype=torch.float32, requires_grad=True)
    route_scores = functional.softplus(reference_logits).sqrt()  # pylint: disable=not-callable
    expert_indices = torch.topk(route_scores, top_k, dim=-1, sorted=False).indices
    sorted_indices = expert_indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all().item())
    selected_scores = route_scores.gather(1, expert_indices)
    denominator = selected_scores.sum(dim=-1, keepdim=True) + 1.0e-20
    routing_weights = selected_scores / denominator * _SCALING
    routing_gradient = torch.tensor(
        ((0.75, -1.25, 2.0), (-0.5, 1.5, -2.25)),
        device=_DEVICE,
        dtype=torch.float32,
    )
    expected_score_grad = torch.autograd.grad(
        routing_weights,
        route_scores,
        grad_outputs=routing_gradient,
        retain_graph=True,
    )[0]
    scaled_gradient = routing_gradient * _SCALING
    expected_selected_grad = scaled_gradient / denominator
    expected_selected_grad -= (selected_scores * scaled_gradient).sum(dim=-1, keepdim=True) / denominator.square()
    scatter_reference = torch.zeros_like(route_scores).scatter_add(1, expert_indices, expected_selected_grad)
    torch.testing.assert_close(scatter_reference, expected_score_grad, rtol=1.0e-5, atol=1.0e-6)
    expected = torch.autograd.grad(routing_weights, reference_logits, grad_outputs=routing_gradient)[0]

    gate = _make_gate()
    plan = gate._plan(reference_logits)  # pylint: disable=protected-access

    def _run_route_grad() -> torch.Tensor:
        with torch.no_grad():
            result = mega_gate_ops._mega_gate_route_grad(  # pylint: disable=protected-access
                reference_logits.detach(),
                route_scores.detach(),
                selected_scores.detach(),
                denominator.detach(),
                expert_indices.detach(),
                routing_gradient,
                reference_logits.detach(),
                plan.grad_route_runtime.normal_tensor,
                plan.grad_route_runtime.normal_tensor,
                top_k=top_k,
                routed_scaling_factor=_SCALING,
                has_direct_grad=False,
            )
        torch.npu.synchronize()
        return result

    actual = _run_route_grad()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(expected)
    close = torch.isclose(actual, expected, rtol=1.0e-5, atol=1.0e-6, equal_nan=True)
    if not bool(close.all().item()):
        softplus_derivative = torch.where(
            reference_logits > 20.0,
            torch.ones_like(reference_logits),
            torch.sigmoid(reference_logits),
        )
        recovered_score_grad = actual * (2.0 * route_scores.detach()) / softplus_derivative
        mismatch = (~close).nonzero().cpu().tolist()
        raise AssertionError(
            "HyperMegaGate RouteGrad mismatch: "
            f"positions={mismatch}, "
            f"expert_indices={expert_indices.cpu().tolist()}, "
            f"actual_dlogits={actual.cpu().tolist()}, "
            f"expected_dlogits={expected.cpu().tolist()}, "
            f"recovered_score_grad={recovered_score_grad.cpu().tolist()}, "
            f"expected_score_grad={expected_score_grad.cpu().tolist()}"
        )
    torch.testing.assert_close(actual[finite], expected[finite], rtol=1.0e-5, atol=1.0e-6)
