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
"""Launch single-card precision checks for the MegaGate module entry."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_gate.py")


def _run_worker(case: str) -> None:
    """Run one MegaGate module worker on one NPU."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaGate ST requires a wheel or PYTHONPATH payload built with --multicore on")
    with without_inherited_rank_environment():
        parallel_run([TorchCase(_WORKER, case, num_proc=1)], global_num_proc=1)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_module_parity() -> None:
    """Compare the formal module entry with golden on 2K, 4K, and 8K."""
    _run_worker("test_mega_gate_module_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_vision_module_parity() -> None:
    """Require native vision Route parity for text, vision, and mixed tokens."""
    _run_worker("test_mega_gate_vision_module_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_module_repeat_determinism() -> None:
    """Require repeated formal-entry executions to be bitwise stable."""
    _run_worker("test_mega_gate_module_repeat_determinism")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_pipeline_profile() -> None:
    """Require the formal entry to export ten stages for every active AIV."""
    _run_worker("test_mega_gate_pipeline_profile")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_grad_pipeline_profile() -> None:
    """Require RouteGrad to export its configured stages per active AIV."""
    _run_worker("test_mega_gate_grad_pipeline_profile")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_module_support_contract() -> None:
    """Require the formal module to accept only its measured forward contract."""
    _run_worker("test_mega_gate_module_support_contract")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_dynamic_configuration_parity() -> None:
    """Require forward parity for runtime dimensions and TopK."""
    _run_worker("test_mega_gate_dynamic_configuration_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_backward_parity() -> None:
    """Require FP32 RouteGrad and final BF16 projection-gradient parity."""
    _run_worker("test_mega_gate_backward_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_vision_backward_parity() -> None:
    """Require 2K/8K mixed-token vision backward parity, including FP32 RouteGrad."""
    _run_worker("test_mega_gate_vision_backward_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_vision_weights_only_fresh_graph_regression() -> None:
    """Reproduce the 8K mixed-vision weights-only failure without profiling."""
    _run_worker("test_mega_gate_vision_weights_only_fresh_graph_regression")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_dynamic_backward_parity() -> None:
    """Require K=1 dynamic-shape and retained-graph backward parity."""
    _run_worker("test_mega_gate_dynamic_backward_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_route_grad_batch_parity() -> None:
    """Cover independent elementwise batches, long sequences, and tail rows."""
    _run_worker("test_mega_gate_route_grad_batch_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_route_grad_extremes() -> None:
    """Require RouteGrad parity around Softplus threshold and underflow inputs."""
    _run_worker("test_mega_gate_route_grad_extremes")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_dynamic_training_graphs() -> None:
    """Cover changing shapes, multiple layers, and outstanding autograd graphs."""
    _run_worker("test_mega_gate_dynamic_training_graphs")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_topk_padding() -> None:
    """Require valid expert indices at DMA and TopK row-alignment boundaries."""
    _run_worker("test_mega_gate_topk_padding")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_checkpoint_accumulation() -> None:
    """Cover both checkpoint modes, gradient accumulation, and optimizer steps."""
    _run_worker("test_mega_gate_checkpoint_accumulation")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
@pytest.mark.parametrize("scenario", ("2k-8k", "8k-32k"))
def test_mega_gate_training_memory_stability(scenario: str) -> None:
    """Report peak memory and check release for standard or long training graphs."""
    _run_worker(f"test_mega_gate_training_memory_stability[{scenario}]")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_gate_dynamic_vision_configuration_parity() -> None:
    """Run the dynamic multimodal worker through the public test launcher."""
    _run_worker("test_mega_gate_dynamic_vision_configuration_parity")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential",
)
def test_mega_gate_state_dict_roundtrip() -> None:
    """Require strict text/vision checkpoint compatibility and training parity."""
    _run_worker("test_mega_gate_state_dict_roundtrip")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential",
)
def test_mega_gate_meta_materialization() -> None:
    """Require meta parameters to materialize through reset or checkpoint loading."""
    _run_worker("test_mega_gate_meta_materialization")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential",
)
def test_mega_gate_autocast_contract() -> None:
    """Check the unchanged training autocast context against the native FP32 contract."""
    _run_worker("test_mega_gate_autocast_contract")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential",
)
def test_mega_gate_correction_bias_update() -> None:
    """Require ranking to observe in-place text and vision bias updates."""
    _run_worker("test_mega_gate_correction_bias_update")


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential",
)
def test_mega_gate_async_training_steps() -> None:
    """Run AdamW accumulation without explicit synchronization between training steps."""
    _run_worker("test_mega_gate_async_training_steps")
