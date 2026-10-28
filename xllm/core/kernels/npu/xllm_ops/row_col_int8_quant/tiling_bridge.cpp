// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
// CANN uses the old libstdc++ ABI; keep it outside the Torch translation unit.
#undef _GLIBCXX_USE_CXX11_ABI
#define _GLIBCXX_USE_CXX11_ABI 0
#include "op_host/row_col_int8_quant_tiling.h"
extern "C" size_t rowcol_make_tiling(uint32_t m,
                                     uint32_t n,
                                     uint32_t pad,
                                     uint64_t ffts,
                                     void* buffer,
                                     size_t capacity) {
  optiling::RowColInt8QuantTilingData data;
  data.set_m(m);
  data.set_n(n);
  data.set_row_pad(pad);
  data.set_ffts_addr(ffts);
  data.SaveToBuffer(buffer, capacity);
  return data.GetDataSize();
}
