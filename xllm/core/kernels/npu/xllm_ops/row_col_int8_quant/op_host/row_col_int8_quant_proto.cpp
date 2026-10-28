// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ge {
namespace {
graphStatus infer_shape(gert::InferShapeContext* context) {
  const auto* x = context->GetInputShape(0);
  if (x->GetDimNum() != 2) {
    return GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = *x;
  *context->GetOutputShape(1) = *x;
  context->GetOutputShape(2)->SetDimNum(1);
  context->GetOutputShape(2)->SetDim(0, x->GetDim(0));
  context->GetOutputShape(3)->SetDimNum(1);
  context->GetOutputShape(3)->SetDim(0, x->GetDim(1));
  return GRAPH_SUCCESS;
}
graphStatus infer_dtype(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, DT_INT8);
  context->SetOutputDataType(1, DT_INT8);
  context->SetOutputDataType(2, DT_FLOAT);
  context->SetOutputDataType(3, DT_FLOAT);
  return GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP(RowColInt8Quant).InferShape(infer_shape).InferDataType(infer_dtype);
}  // namespace ge
