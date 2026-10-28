// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RowColInt8QuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, m);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, row_pad);
TILING_DATA_FIELD_DEF(uint64_t, ffts_addr);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RowColInt8Quant, RowColInt8QuantTilingData)
}  // namespace optiling
