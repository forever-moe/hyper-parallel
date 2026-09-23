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
"""Lightweight model-facing MegaGate module."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.modules.mega_gate.function import mega_gate
from hyper_parallel.core.multicore.modules.mega_gate.plan import MegaGatePlan, build_mega_gate_plan


@lru_cache(maxsize=None)
def _plan_for_device(device_index: int) -> MegaGatePlan:
    """Build one immutable MegaGate plan per process and NPU device."""
    return build_mega_gate_plan(torch.device("npu", device_index))


class MegaGate(torch.nn.Module):
    """DeepSeek V4.1 compatible router with an optional native fast path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        scoring_func: str,
        routed_scaling_factor: float,
        vision_enabled: bool = False,
        initializer_range: float = 0.02,
    ) -> None:
        """Create projection and correction-bias parameters."""
        super().__init__()
        if hidden_size <= 0 or num_experts <= 0:
            raise ValueError("hidden_size and num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if scoring_func not in ("sqrtsoftplus", "softmax", "sigmoid"):
            raise ValueError(f"unsupported scoring_func: {scoring_func!r}")
        if not math.isfinite(routed_scaling_factor):
            raise ValueError("routed_scaling_factor must be finite")
        if not math.isfinite(initializer_range) or initializer_range < 0.0:
            raise ValueError("initializer_range must be finite and nonnegative")

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.scoring_func = scoring_func
        self._native_scoring_supported = scoring_func == "sqrtsoftplus"
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.initializer_range = float(initializer_range)

        self.weight = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        if vision_enabled:
            self.bias_vl = torch.nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_parameter("bias_vl", None)
        self.reset_parameters()

    @classmethod
    def from_config(cls, config: Any) -> "MegaGate":
        """Create a router from the DeepSeek V4.1 configuration fields."""
        return cls(
            hidden_size=int(config.hidden_size),
            num_experts=int(config.num_local_experts),
            top_k=int(config.num_experts_per_tok),
            scoring_func=str(config.scoring_func),
            routed_scaling_factor=float(config.routed_scaling_factor),
            vision_enabled=bool(getattr(config, "v41_vision_enabled", False)),
            initializer_range=float(getattr(config, "initializer_range", 0.02)),
        )

    def reset_parameters(self) -> None:
        """Initialize parameters using the model configuration convention."""
        torch.nn.init.normal_(self.weight, mean=0.0, std=self.initializer_range)
        torch.nn.init.zeros_(self.bias)
        if self.bias_vl is not None:
            torch.nn.init.zeros_(self.bias_vl)

    @staticmethod
    def _device_index(hidden_states: torch.Tensor) -> int:
        device_index = hidden_states.device.index
        return torch.npu.current_device() if device_index is None else device_index

    def _plan(self, hidden_states: torch.Tensor) -> MegaGatePlan:
        """Return the process-level plan for an NPU tensor."""
        return _plan_for_device(self._device_index(hidden_states))

    def _torch_forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the DeepSeek V4.1 reference operation order with Torch ops."""
        flattened = hidden_states.reshape(-1, self.hidden_size)
        logits = functional.linear(  # pylint: disable=not-callable
            flattened.float(), self.weight.float()
        )
        if self.scoring_func == "sqrtsoftplus":
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()

        correction_bias = self.bias
        if image_mask is not None:
            if hidden_states.ndim < 3 or tuple(image_mask.shape) != tuple(hidden_states.shape[:-1]):
                raise ValueError("image_mask must match hidden_states leading dimensions")
            if self.bias_vl is not None:
                correction_bias = torch.where(
                    image_mask.reshape(-1, 1),
                    self.bias_vl.unsqueeze(0),
                    self.bias.unsqueeze(0),
                )
        expert_indices = torch.topk(
            scores + correction_bias,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        routing_weights = scores.gather(1, expert_indices)
        if self.top_k > 1:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1.0e-20
            )
        return logits, routing_weights * self.routed_scaling_factor, expert_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return FP32 logits and weights plus INT64 selected expert indices."""
        if self._native_scoring_supported and hidden_states.device.type == "npu":
            vision_bias = self.bias_vl
            use_vision_bias = image_mask is not None and vision_bias is not None
            flat_image_mask = None
            if image_mask is not None and vision_bias is not None:
                if image_mask.shape != hidden_states.shape[:-1]:
                    raise ValueError("image_mask must match hidden_states leading dimensions")
                flat_image_mask = image_mask.view(-1)
            return mega_gate(
                hidden_states,
                self.weight,
                self.bias,
                self._plan(hidden_states),
                vision_bias=vision_bias if use_vision_bias else None,
                image_mask=flat_image_mask,
                top_k=self.top_k,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        return self._torch_forward(hidden_states, image_mask)


__all__ = ["MegaGate"]
