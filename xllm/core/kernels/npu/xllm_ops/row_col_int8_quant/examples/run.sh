#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
op_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vendor=/opt/asc-op-dev/vendors/rowcol_int8_quant_xllm_math
repo=$(git -C "$op_dir" rev-parse --show-toplevel)
binary="$(dirname "$repo")/state/build/rowcol-example"
g++ -std=c++17 -O2 "$op_dir/examples/aclnn_example.cpp" -I"$ASCEND_HOME_PATH/include" \
  -I"$vendor/op_api/include/aclnnop" -L"$vendor/op_api/lib" -lcust_opapi \
  -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lnnopbase -Wl,-rpath,"$vendor/op_api/lib" -o "$binary"
"$binary"
