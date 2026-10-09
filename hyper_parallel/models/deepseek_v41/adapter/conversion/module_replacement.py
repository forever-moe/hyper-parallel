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
"""Identity-preserving MegaGate replacement for the DeepSeek V4.1 router."""

from collections.abc import Mapping
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def replace_deepseek_v41_mega_gate(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> MegaGate:
    """Replace a V4.1 router without materializing or copying its parameters.

    Args:
        module: Original router, including CPU, NPU or meta parameters.
        module_fqn: Model path used for validation diagnostics.
        context: Builder replacement context; no parallel resources are allocated.

    Returns:
        MegaGate sharing the original parameters and preserving its training state.
        Native libraries and per-device plans are loaded on the first NPU call.

    Raises:
        TypeError: The source is not a DeepseekV41TopKRouter.
    """
    del context
    if not isinstance(module, DeepseekV41TopKRouter):
        raise TypeError(f"{module_fqn}: MegaGate replacement requires DeepseekV41TopKRouter")
    # Meta construction avoids consuming RNG or allocating throwaway projection weights.
    with torch.device("meta"):
        replacement = MegaGate(
            hidden_size=module.hidden_size,
            num_experts=module.num_experts,
            top_k=module.top_k,
            scoring_func=module.scoring_func,
            routed_scaling_factor=module.routed_scaling_factor,
            vision_enabled=module.bias_vl is not None,
        )
    replacement.weight = module.weight
    replacement.bias = module.bias
    replacement.bias_vl = module.bias_vl
    replacement.train(module.training)
    return replacement


__all__ = ["replace_deepseek_v41_mega_gate"]
