// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "common.hpp"

namespace ttnn::prim {

struct ReduceParams {
    tt::tt_metal::ReduceOpMath math_op{};
    tt::tt_metal::ReduceOpDim dim{};
    float scaler{1.0f};  // runtime arg; excluded from the program hash
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    bool negate{false};
    float post_mul_scaler{1.0f};  // runtime arg; excluded from the program hash. live when scaler_mode == PostMul
    ScalerMode scaler_mode{ScalerMode::ScalerTile};
    // Dense RM path for mean (AVG) and sum: 4D BF16/FLOAT32 interleaved I/O. AVG is lowered
    // to SUM + scaler before launch. Other ROW_MAJOR reductions tilize. Exactly one flag
    // may be set (validated in validate_on_program_cache_miss).
    bool row_major_w_dense_path{false};
    bool row_major_h_dense_path{false};
    // Accurate fp32: route Float32 through the SFPU (full fp32); set from
    // fast_and_approximate_mode=False on sum/mean/max/min.
    bool use_sfpu_reduce{false};
    // Number of contiguous H segments to reduce independently (RM-H dense path; 1 = no split).
    // Spreads a tall-H reduce over more cores, yielding a (N, C, num_h_slices, W) partial.
    uint32_t num_h_slices{1};
    // Physical layout the op must produce: TILE on the tilized paths, ROW_MAJOR on the dense RM
    // ones, or TILE from RM-H when num_h_slices == 1.
    tt::tt_metal::Layout output_layout{tt::tt_metal::Layout::TILE};
};

}  // namespace ttnn::prim
