// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace row_col_int8_quant {
inline bool use_tiled(uint32_t m, uint32_t /*n*/) { return m >= 512; }
inline uint64_t workspace_bytes(uint32_t m, uint32_t n) {
  return use_tiled(m, n) ? 16ULL * 1024 * 1024 +
                               4ULL * ((m + 63) / 64) * ((n + 255) / 256) * 320
                         : 0;
}
}  // namespace row_col_int8_quant
