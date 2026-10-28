// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include "row_col_int8_quant_tiling.h"

#include <algorithm>

#include "acl/acl_rt.h"
#include "plan.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
namespace {
ge::graphStatus tile(gert::TilingContext* context) {
  const auto& shape = context->GetInputShape(0)->GetOriginShape();
  if (shape.GetDimNum() != 2) {
    return ge::GRAPH_FAILED;
  }
  int64_t m = shape.GetDim(0), n = shape.GetDim(1);
  if (m < 1 || m > 4096 || n < 2 || n > 4096 || n % 2 != 0) {
    return ge::GRAPH_FAILED;
  }
  RowColInt8QuantTilingData data;
  data.set_m(m);
  data.set_n(n);
  uint32_t pad = 64;
  while (pad < n) {
    pad *= 2;
  }
  data.set_row_pad(pad);
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  context->SetBlockDim(std::min<uint32_t>(platform.GetCoreNumAiv(),
                                          std::max<int64_t>(m, (n + 63) / 64)));
  bool tiled = row_col_int8_quant::use_tiled(m, n);
  void* hardware_sync = nullptr;
  if (tiled && aclrtGetHardwareSyncAddr(&hardware_sync) != ACL_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  data.set_ffts_addr(reinterpret_cast<uint64_t>(hardware_sync));
  if (tiled) {
    context->SetScheduleMode(1);
  }
  context->SetTilingKey(tiled ? 1 : 0);
  context->GetWorkspaceSizes(1)[0] = row_col_int8_quant::workspace_bytes(m, n);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(RowColInt8Quant).Tiling(tile);
}  // namespace optiling
