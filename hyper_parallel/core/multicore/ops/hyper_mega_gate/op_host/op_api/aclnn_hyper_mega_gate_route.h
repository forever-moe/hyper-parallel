/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef ACLNN_HYPER_MEGA_GATE_ROUTE_H
#define ACLNN_HYPER_MEGA_GATE_ROUTE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnHyperMegaGateRouteGetWorkspaceSize(
  const aclTensor *logits, const aclTensor *text_bias, const aclTensor *vision_bias, const aclTensor *image_mask,
  const aclTensor *runtime_config, const aclTensor *profile_buffer, const aclTensor *routing_weights,
  const aclTensor *expert_indices, const aclTensor *route_scores, const aclTensor *selected_scores,
  const aclTensor *normalization_denominator, int64_t top_k, double routed_scaling_factor, bool use_vision_bias,
  uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnHyperMegaGateRoute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_HYPER_MEGA_GATE_ROUTE_H
