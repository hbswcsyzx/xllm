# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(row_col_int8_quant LANGUAGES CXX)
# Delegate to the pinned xllm_ops packaging framework rather than duplicating it.
add_custom_target(rowcol_package
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  USES_TERMINAL)
add_custom_target(rowcol_bridge
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/build_bridge.py
  DEPENDS rowcol_package
  USES_TERMINAL)
add_custom_target(rowcol_example
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/examples/run.sh
  DEPENDS rowcol_package
  USES_TERMINAL)
