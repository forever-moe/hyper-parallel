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
"""Export HyperMegaGate's dynamic-AIV internal pipeline as a Chrome trace."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# pylint: disable=wrong-import-position
import torch  # pylint: disable=forbidden-backend-import
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_gate import MegaGate
from hyper_parallel.core.multicore import profiler as multicore_profiler
# pylint: enable=wrong-import-position


def main() -> int:
    """Capture 2K, 4K, or 8K V4.1 forward on the selected NPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(2048, 4096, 8192), default=4096)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--route-mode", choices=("vision-mixed", "vision", "text"), default="vision-mixed")
    parser.add_argument("--vision-token-ratio", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.route_mode == "vision-mixed" and not 0.0 < args.vision_token_ratio < 1.0:
        parser.error("vision-token-ratio must be between zero and one for vision-mixed")
    torch.npu.set_device(args.device_index)
    device = torch.device("npu", args.device_index)
    hidden = torch.randn((1, args.tokens, 5120), device=device, dtype=torch.bfloat16)
    weight = torch.randn((384, 5120), device=device, dtype=torch.bfloat16)
    bias = torch.zeros((384,), device=device, dtype=torch.float32)
    bias_vl = torch.linspace(0.5, -0.5, 384, device=device, dtype=torch.float32)
    if args.route_mode == "text":
        image_mask = None
    else:
        vision_tokens = args.tokens if args.route_mode == "vision" else round(args.tokens * args.vision_token_ratio)
        image_mask = (torch.arange(args.tokens, device=device) < vision_tokens).reshape(1, args.tokens)
    gate = MegaGate(
        hidden_size=5120,
        num_experts=384,
        scoring_func="sqrtsoftplus",
        top_k=6,
        routed_scaling_factor=1.5,
        vision_enabled=args.route_mode != "text",
    )
    gate.weight = torch.nn.Parameter(weight, requires_grad=False)
    gate.bias = torch.nn.Parameter(bias, requires_grad=False)
    if gate.bias_vl is not None:
        gate.bias_vl = torch.nn.Parameter(bias_vl, requires_grad=False)

    def call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch one HyperMegaGate forward on the current stream."""
        return gate(hidden, image_mask=image_mask)

    with torch.no_grad():
        for _ in range(args.warmup):
            call()
        torch.npu.synchronize()
        with multicore_profiler.mega_kernel_profile(detailed_task_names=True) as profiler:
            call()
            profiler.step()
        trace = profiler.export_chrome_trace(args.output)

    metadata = trace["megaKernelCycleTrace"]
    compute_events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    tracks = {event["tid"] for event in compute_events}
    result = {
        "trace_path": str(args.output),
        "route_mode": args.route_mode,
        "vision_tokens": 0 if image_mask is None else int(image_mask.sum().item()),
        "invocation_count": metadata["invocationCount"],
        "record_count": metadata["recordCount"],
        "aiv_track_count": len(tracks),
        "dropped_record_count": metadata["droppedRecordCount"],
        "warnings": metadata["warnings"],
    }
    valid = (
        result["invocation_count"] == 1
        and result["aiv_track_count"] > 0
        and result["record_count"] == result["aiv_track_count"] * 10
        and result["dropped_record_count"] == 0
        and result["warnings"] == []
    )
    if not valid:
        raise RuntimeError(f"unexpected HyperMegaGate internal trace: {result}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
