// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class RowColInt8Quant final : public OpDef {
 public:
  explicit RowColInt8Quant(const char* name) : OpDef(name) {
    Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    Output("qRow")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    Output("qCol")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    Output("rowAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    Output("colAbsmax")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(RowColInt8Quant);
}  // namespace ops
