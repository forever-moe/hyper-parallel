/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H
#define ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Prepare the fused normalization gradient followed by production CANN
 * LinearIndex, ScatterElementsV2, Muls, RealDiv, SoftplusV2Grad, and the
 * optional combined Add.
 */
ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGradGetWorkspaceSize(
  const aclTensor *logits, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, const aclTensor *expert_indices, const aclTensor *grad_routing_weights,
  const aclTensor *direct_grad_logits, const aclTensor *runtime_config, const aclTensor *profile_buffer,
  const aclTensor *grad_logits, int64_t top_k, double routed_scaling_factor, bool has_direct_grad,
  uint64_t *workspaceSize, aclOpExecutor **executor);

/** Enqueue the prepared Route backward operation; completion follows stream order. */
ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_HYPER_MEGA_GATE_ROUTE_GRAD_H
