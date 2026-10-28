// Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

namespace {
__aicore__ inline void mask_count(uint32_t count) {
  set_mask_count();
  set_vector_mask(0, count);
}
__aicore__ inline void mask_full() {
  set_mask_norm();
  set_vector_mask(0, ~0ULL);
}
__aicore__ inline void global_ready(uint16_t flag) {
  ffts_cross_core_sync(PIPE_MTE3, static_cast<uint16_t>(1 + (flag << 8)));
  wait_flag_dev(flag);
}
__aicore__ inline void finish_quant(__ubuf__ float* f,
                                    __ubuf__ float* temp,
                                    __ubuf__ half* h) {
  mask_count(16384);
  vmaxs(f, f, -127.0f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vmins(f, f, 127.0f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322s32r(reinterpret_cast<__ubuf__ int32_t*>(temp), f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_s322f32(f, reinterpret_cast<__ubuf__ int32_t*>(temp), 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(h, f, 1, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
  vconv_f162s8(reinterpret_cast<__ubuf__ int8_t*>(temp), h, 1, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
  mask_full();
}
__aicore__ inline void tiled(GM_ADDR x,
                             GM_ADDR qr,
                             GM_ADDR qc,
                             GM_ADDR r,
                             GM_ADDR c,
                             GM_ADDR workspace,
                             uint32_t m,
                             uint32_t n) {
  uint32_t id = get_block_idx(), cores = get_block_num();
  uint32_t mt = (m + 63) / 64, nt = (n + 255) / 256;
  auto* h = reinterpret_cast<__ubuf__ half*>(0);
  auto* f = reinterpret_cast<__ubuf__ float*>(32768);
  auto* a = reinterpret_cast<__ubuf__ float*>(98304);
  auto* rs = reinterpret_cast<__ubuf__ float*>(163840);
  auto* cs = reinterpret_cast<__ubuf__ float*>(164096);
  auto* ones = reinterpret_cast<__ubuf__ float*>(165120);
  auto* rb = reinterpret_cast<__ubuf__ float*>(166144);
  auto* row_partial = reinterpret_cast<__gm__ float*>(workspace);
  auto* col_partial = row_partial + mt * nt * 64;
  mask_full();
  set_mov_pad_val(0);
  for (uint32_t tile = id; tile < mt * nt; tile += cores) {
    uint32_t ri = tile / nt, ci = tile % nt;
    uint32_t rows = bounded(64, m - ri * 64), cols = bounded(256, n - ci * 256);
    mask_count(16384);
    vector_dup(h, static_cast<half>(0), 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    mask_full();
    v_to_mte2();
    copy_gm_to_ubuf_align_b16(
        h,
        reinterpret_cast<__gm__ half*>(x) + ri * 64 * n + ci * 256,
        0,
        rows,
        cols * 2,
        0,
        0,
        (n - cols) * 2,
        (256 - (cols + 15) / 16 * 16) / 16);
    mte2_to_v();
    mask_count(16384);
    vconv_f162f32(f, h, 1, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    vabs(a, f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    mask_full();
    vmax(a, a, a + 64, 64, 1, 1, 1, 32, 32, 32);
    pipe_barrier(PIPE_V);
    vmax(a, a, a + 128, 64, 1, 1, 1, 32, 32, 32);
    pipe_barrier(PIPE_V);
    vmax(a, a, a + 192, 64, 1, 1, 1, 32, 32, 32);
    pipe_barrier(PIPE_V);
    vcmax(rs, a, 64, 1, 1, 32, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    // Re-create abs: row reduction overwrote the first 64 columns.
    mask_count(16384);
    vabs(a, f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    for (uint32_t half_size = 8192; half_size >= 256; half_size /= 2) {
      mask_count(half_size);
      vmax(a, a, a + half_size, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    mask_full();
    v_to_mte3();
    copy_ubuf_to_gm_align_b32(
        row_partial + tile * 64, rs, 0, 1, 256, 0, 0, 0, 0);
    copy_ubuf_to_gm_align_b32(
        col_partial + (ci * mt + ri) * 256, a, 0, 1, 1024, 0, 0, 0, 0);
    mte3_to_v();
  }
  global_ready(14);
  // Grouped contiguous partial arrays avoid a scalar DMA per partial.
  for (uint32_t ri = id; ri < mt; ri += cores) {
    v_to_mte2();
    copy_gm_to_ubuf(f, row_partial + ri * nt * 64, 0, 1, nt * 8, 0, 0);
    mte2_to_v();
    vector_dup(rs, 0.0f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    for (uint32_t ci = 0; ci < nt; ++ci) {
      vmax(rs, rs, f + ci * 64, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    v_to_mte3();
    copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float*>(r) + ri * 64,
                              rs,
                              0,
                              1,
                              bounded(64, m - ri * 64) * 4,
                              0,
                              0,
                              0,
                              0);
    mte3_to_v();
  }
  for (uint32_t ci = id; ci < nt; ci += cores) {
    v_to_mte2();
    copy_gm_to_ubuf(f, col_partial + ci * mt * 256, 0, 1, mt * 32, 0, 0);
    mte2_to_v();
    // An arbitrary mt is reduced with a fixed accumulator and exact loop bound.
    mask_count(256);
    vector_dup(cs, 0.0f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    for (uint32_t ri = 0; ri < mt; ++ri) {
      vmax(cs, cs, f + ri * 256, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    mask_full();
    v_to_mte3();
    copy_ubuf_to_gm_align_b32(reinterpret_cast<__gm__ float*>(c) + ci * 256,
                              cs,
                              0,
                              1,
                              bounded(256, n - ci * 256) * 4,
                              0,
                              0,
                              0,
                              0);
    mte3_to_v();
  }
  global_ready(15);
  for (uint32_t tile = id; tile < mt * nt; tile += cores) {
    uint32_t ri = tile / nt, ci = tile % nt;
    uint32_t rows = bounded(64, m - ri * 64), cols = bounded(256, n - ci * 256);
    v_to_mte2();
    copy_gm_to_ubuf_align_b16(
        h,
        reinterpret_cast<__gm__ half*>(x) + ri * 64 * n + ci * 256,
        0,
        rows,
        cols * 2,
        0,
        0,
        (n - cols) * 2,
        (256 - (cols + 15) / 16 * 16) / 16);
    copy_gm_to_ubuf_align_b32(rs,
                              reinterpret_cast<__gm__ float*>(r) + ri * 64,
                              0,
                              1,
                              rows * 4,
                              0,
                              0,
                              0,
                              0);
    copy_gm_to_ubuf_align_b32(cs,
                              reinterpret_cast<__gm__ float*>(c) + ci * 256,
                              0,
                              1,
                              cols * 4,
                              0,
                              0,
                              0,
                              0);
    mte2_to_v();
    mask_count(256);
    vector_dup(ones, 127.0f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vmaxs(cs, cs, 0x1p-24f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(cs, ones, cs, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    mask_full();
    vmaxs(rs, rs, 0x1p-24f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(rs, ones, rs, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vbrcb(reinterpret_cast<__ubuf__ uint32_t*>(rb),
          reinterpret_cast<__ubuf__ uint32_t*>(rs),
          1,
          8,
          8);
    pipe_barrier(PIPE_V);
    mask_count(16384);
    vconv_f162f32(f, h, 1, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    mask_full();
    for (uint32_t offset = 0; offset < 256; offset += 64) {
      vmul(a + offset, f + offset, rb, 64, 1, 1, 0, 32, 32, 1);
    }
    pipe_barrier(PIPE_V);
    // Quantize row values in a; scratch is f, half conversion aliases h.
    finish_quant(a, f, h);
    v_to_mte3();
    copy_ubuf_to_gm_align_b8(
        reinterpret_cast<__gm__ int8_t*>(qr) + ri * 64 * n + ci * 256,
        reinterpret_cast<__ubuf__ int8_t*>(f),
        0,
        rows,
        cols,
        0,
        0,
        (256 - (cols + 31) / 32 * 32) / 32,
        n - cols);
    mte3_to_v();
    // h was consumed by rounding. Reload original input for the independent
    // column result.
    v_to_mte2();
    copy_gm_to_ubuf_align_b16(
        h,
        reinterpret_cast<__gm__ half*>(x) + ri * 64 * n + ci * 256,
        0,
        rows,
        cols * 2,
        0,
        0,
        (n - cols) * 2,
        (256 - (cols + 15) / 16 * 16) / 16);
    mte2_to_v();
    mask_count(16384);
    vconv_f162f32(f, h, 1, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    mask_full();
    for (uint32_t offset = 0; offset < 256; offset += 64) {
      vmul(a + offset, f + offset, cs + offset, 64, 1, 1, 1, 32, 32, 0);
    }
    pipe_barrier(PIPE_V);
    finish_quant(a, f, h);
    v_to_mte3();
    copy_ubuf_to_gm_align_b8(
        reinterpret_cast<__gm__ int8_t*>(qc) + ri * 64 * n + ci * 256,
        reinterpret_cast<__ubuf__ int8_t*>(f),
        0,
        rows,
        cols,
        0,
        0,
        (256 - (cols + 31) / 32 * 32) / 32,
        n - cols);
    mte3_to_v();
  }
}
}  // namespace
