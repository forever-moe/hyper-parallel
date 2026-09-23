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
"""Torch-only Multicore APIs, separate from the HyperParallel root exports."""

from hyper_parallel.core.multicore import profiler
from hyper_parallel.core.multicore.modules.mega_gate.module import MegaGate

__all__ = ["MegaGate", "MegaMoeExperts", "profiler"]


def __getattr__(name: str):
    """Load the optional NPU MoE backend only when its public module is requested."""
    if name == "MegaMoeExperts":
        # MegaGate construction and CPU/meta model initialization need no torch_npu.
        from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts

        globals()[name] = MegaMoeExperts
        return MegaMoeExperts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
