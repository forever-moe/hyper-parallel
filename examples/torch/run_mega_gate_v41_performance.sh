#!/bin/bash
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

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 OUTPUT_DIR" >&2
    echo "Example: $0 ../workspace/prof_0923_route_grad_11stage" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR=$1
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE_INDEX=${DEVICE_INDEX:-0}
ROUTE_MODE=${ROUTE_MODE:-vision-mixed}
VISION_TOKEN_RATIO=${VISION_TOKEN_RATIO:-0.25}
WARMUP=${WARMUP:-10}
PROFILE_ITERATIONS=${PROFILE_ITERATIONS:-20}
WALL_ITERATIONS=${WALL_ITERATIONS:-500}
WALL_REPEATS=${WALL_REPEATS:-5}
QUICK=${QUICK:-0}
TOKENS=${TOKENS:-}

COMMON_ARGS=(
    --device-index "${DEVICE_INDEX}"
    --route-mode "${ROUTE_MODE}"
    --vision-token-ratio "${VISION_TOKEN_RATIO}"
    --warmup "${WARMUP}"
    --iterations "${PROFILE_ITERATIONS}"
    --wall-iterations "${WALL_ITERATIONS}"
    --wall-repeats "${WALL_REPEATS}"
)
if [[ "${QUICK}" == "1" ]]; then
    if [[ -n "${TOKENS}" ]]; then
        echo "QUICK=1 and TOKENS cannot be used together" >&2
        exit 2
    fi
    COMMON_ARGS+=(--quick)
elif [[ -n "${TOKENS}" ]]; then
    read -r -a TOKEN_COUNTS <<< "${TOKENS}"
    COMMON_ARGS+=(--tokens "${TOKEN_COUNTS[@]}")
fi

# Blocking launch changes the measured execution model and must stay disabled.
unset ASCEND_LAUNCH_BLOCKING

cd "${PROJECT_ROOT}"

failed=0
run_benchmark() {
    local name=$1
    shift
    echo "===== ${name} ====="
    if "$@"; then
        echo "===== ${name}: completed ====="
    else
        local status=$?
        echo "===== ${name}: failed with status ${status}; continuing =====" >&2
        failed=1
    fi
}

run_benchmark forward \
    "${PYTHON_BIN}" examples/torch/mega_gate_v41_forward_benchmark.py \
    --output "${OUTPUT_DIR}/forward.json" \
    --trace-dir "${OUTPUT_DIR}/forward_traces" \
    "${COMMON_ARGS[@]}"

run_benchmark backward \
    "${PYTHON_BIN}" examples/torch/mega_gate_v41_backward_benchmark.py \
    --output "${OUTPUT_DIR}/backward.json" \
    --trace-dir "${OUTPUT_DIR}/backward_traces" \
    --mk-trace-dir "${OUTPUT_DIR}/backward_mk_traces" \
    "${COMMON_ARGS[@]}"

run_benchmark forward_backward \
    "${PYTHON_BIN}" examples/torch/mega_gate_v41_backward_benchmark.py \
    --include-forward \
    --output "${OUTPUT_DIR}/forward_backward.json" \
    --trace-dir "${OUTPUT_DIR}/forward_backward_traces" \
    --mk-trace-dir "${OUTPUT_DIR}/forward_backward_mk_traces" \
    "${COMMON_ARGS[@]}"

echo "Results: ${OUTPUT_DIR}"
exit "${failed}"
