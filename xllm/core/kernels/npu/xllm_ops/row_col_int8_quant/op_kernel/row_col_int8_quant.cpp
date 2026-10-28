// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
__aicore__ inline uint32_t bounded(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}
__aicore__ inline void mte2_to_v() {
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}
__aicore__ inline void v_to_mte2() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}
__aicore__ inline void v_to_mte3() {
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
}
__aicore__ inline void mte3_to_v() {
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}
__aicore__ inline void quantize(__ubuf__ float* f, uint32_t count) {
  auto* ints = reinterpret_cast<__ubuf__ int32_t*>(81920);
  auto* h = reinterpret_cast<__ubuf__ half*>(114688);
  auto* q = reinterpret_cast<__ubuf__ int8_t*>(131072);
  uint32_t repeats = count / 64;
  vmaxs(f, f, -127.0f, repeats, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vmins(f, f, 127.0f, repeats, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322s32r(ints, f, repeats, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_s322f32(f, ints, repeats, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(h, f, repeats, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
  vconv_f162s8(q, h, repeats, 1, 1, 2, 4);
  pipe_barrier(PIPE_V);
}
}  // namespace

#include "tiled.h"

extern "C" __global__ __aicore__ void row_col_int8_quant(GM_ADDR x,
                                                         GM_ADDR qr,
                                                         GM_ADDR qc,
                                                         GM_ADDR r,
                                                         GM_ADDR c,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0);
  GET_TILING_DATA(t, tiling);
  if constexpr (TILING_KEY_IS(1)) {
    set_ffts_base_addr(t.ffts_addr);
    tiled(x, qr, qc, r, c, workspace, t.m, t.n);
    return;
  }
  if constexpr (TILING_KEY_IS(0)) {
    const uint32_t m = t.m, n = t.n, pad = t.row_pad;
    const uint32_t id = get_block_idx(), cores = get_block_num();
    auto* input = reinterpret_cast<__ubuf__ half*>(0);
    auto* f = reinterpret_cast<__ubuf__ float*>(16384);
    auto* a = reinterpret_cast<__ubuf__ float*>(49152);
    auto* q = reinterpret_cast<__ubuf__ int8_t*>(131072);
    auto* stat = reinterpret_cast<__ubuf__ float*>(139264);
    auto* scale = reinterpret_cast<__ubuf__ float*>(139520);
    auto* ones = reinterpret_cast<__ubuf__ float*>(139776);
    set_mask_norm();
    set_vector_mask(0, ~0ULL);
    set_mov_pad_val(0);
    vector_dup(ones, 127.0f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    const uint32_t row_batch = bounded(4, 8192 / pad);
    const uint32_t per_core = (m + cores - 1) / cores;
    const uint32_t row_end = bounded(m, (id + 1) * per_core);
    for (uint32_t row = id * per_core; row < row_end; row += row_batch) {
      const uint32_t rows = bounded(row_batch, row_end - row);
      const uint32_t count = rows * pad;
      vector_dup(input, static_cast<half>(0), count / 64, 1, 1, 4, 0);
      pipe_barrier(PIPE_V);
      v_to_mte2();
      copy_gm_to_ubuf_align_b16(input,
                                reinterpret_cast<__gm__ half*>(x) + row * n,
                                0,
                                rows,
                                n * 2,
                                0,
                                0,
                                0,
                                (pad - (n + 15) / 16 * 16) / 16);
      mte2_to_v();
      vconv_f162f32(f, input, count / 64, 1, 1, 8, 4);
      pipe_barrier(PIPE_V);
      vabs(a, f, count / 64, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        auto* absolute_row = a + local_row * pad;
        for (uint32_t half_size = pad / 2; half_size >= 64; half_size /= 2) {
          vmax(absolute_row,
               absolute_row,
               absolute_row + half_size,
               half_size / 64,
               1,
               1,
               1,
               8,
               8,
               8);
          pipe_barrier(PIPE_V);
        }
        vcmax(stat + local_row * 8,
              absolute_row,
              1,
              1,
              1,
              8,
              Order_t::ONLY_VALUE);
      }
      pipe_barrier(PIPE_V);
      v_to_mte3();
      copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float*>(r) + row,
                                stat,
                                0,
                                rows,
                                4,
                                0,
                                0,
                                0,
                                0);
      vmaxs(scale, stat, 0x1p-24f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(scale, ones, scale, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        float ratio = scale[local_row * 8];
        vmuls(f + local_row * pad,
              f + local_row * pad,
              ratio,
              pad / 64,
              1,
              1,
              8,
              8);
      }
      pipe_barrier(PIPE_V);
      quantize(f, count);
      v_to_mte3();
      copy_ubuf_to_gm_align_b8(reinterpret_cast<__gm__ int8_t*>(qr) + row * n,
                               q,
                               0,
                               rows,
                               n,
                               0,
                               0,
                               (pad - (n + 31) / 32 * 32) / 32,
                               0);
      mte3_to_v();
    }
    for (uint32_t col = id * 64; col < n; col += cores * 64) {
      uint32_t width = bounded(64, n - col);
      vector_dup(stat, 0.0f, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < m; row += 128) {
        uint32_t rows = bounded(128, m - row);
        vector_dup(input, static_cast<half>(0), 128, 1, 1, 4, 0);
        pipe_barrier(PIPE_V);
        v_to_mte2();
        copy_gm_to_ubuf_align_b16(
            input,
            reinterpret_cast<__gm__ half*>(x) + row * n + col,
            0,
            rows,
            width * 2,
            0,
            0,
            (n - width) * 2,
            (64 - (width + 15) / 16 * 16) / 16);
        mte2_to_v();
        vconv_f162f32(f, input, 128, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vabs(a, f, 128, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        for (uint32_t half_rows = 64; half_rows > 0; half_rows /= 2) {
          vmax(a, a, a + half_rows * 64, half_rows, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
        }
        vmax(stat, stat, a, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
      }
      v_to_mte3();
      copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float*>(c) + col,
                                stat,
                                0,
                                1,
                                width * 4,
                                0,
                                0,
                                0,
                                0);
      vmaxs(scale, stat, 0x1p-24f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vdiv(scale, ones, scale, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < m; row += 128) {
        uint32_t rows = bounded(128, m - row);
        v_to_mte2();
        copy_gm_to_ubuf_align_b16(
            input,
            reinterpret_cast<__gm__ half*>(x) + row * n + col,
            0,
            rows,
            width * 2,
            0,
            0,
            (n - width) * 2,
            (64 - (width + 15) / 16 * 16) / 16);
        mte2_to_v();
        vconv_f162f32(f, input, 128, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(f, f, scale, 128, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        quantize(f, 8192);
        v_to_mte3();
        copy_ubuf_to_gm_align_b8(
            reinterpret_cast<__gm__ int8_t*>(qc) + row * n + col,
            q,
            0,
            rows,
            width,
            0,
            0,
            (64 - (width + 31) / 32 * 32) / 32,
            n - width);
        mte3_to_v();
      }
    }
  }
}
