#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
op_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$op_dir" rev-parse --show-toplevel)
task_root=$(dirname "$repo")
stage="$task_root/state/build/rowcol-framework"
build="$task_root/state/build/rowcol-aclnn"
exec 9>"$task_root/state/build/rowcol-build.lock"
flock 9
mkdir -p "$stage"
for part in cmake common scripts utils; do
  if [[ ! -e "$stage/$part" ]]; then cp -a "$repo/third_party/xllm_ops/$part" "$stage/"; fi
done
for file in "$repo/third_party/xllm_ops/xllm_ops/"*; do
  if [[ -f "$file" ]]; then cp --remove-destination "$file" "$stage/xllm_ops/"; fi
done
mkdir -p "$stage/xllm_ops/row_col_int8_quant"
cp -a "$op_dir/op_host" "$op_dir/op_kernel" "$stage/xllm_ops/row_col_int8_quant/"
# Framework configure snapshots kernel sources; invalidate only this task's binaries.
rm -rf "$build/binary"
cd "$stage/xllm_ops"
bash build.sh --pkg --ops=row_col_int8_quant --soc=ascend910_93 --vendor_name=rowcol_int8_quant --build-dir="$build" -j8
package=("$build"/cann-ops-xllm*.run)
bash "${package[0]}" --install-path=/opt/asc-op-dev
mkdir -p "$op_dir/artifacts"
cp "$build"/binary/ascend910_93/bin/row_col_int8_quant/*.o "$op_dir/artifacts/kernel.o"
cp "$build"/binary/ascend910_93/bin/row_col_int8_quant/RowCol*.json "$op_dir/artifacts/kernel.json"
sha256sum "${package[0]}"
