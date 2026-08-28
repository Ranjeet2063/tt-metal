# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for reduction/generic program cache behavior.

Tests target potential caching issues.
The ReduceDeviceOperation uses 3 ProgramFactory variants:
  - ReduceMultiCoreHProgramFactory (dim=H)
  - ReduceMultiCoreWProgramFactory (dim=W)
  - ReduceSingleCoreHwProgramFactory (dim=HW with single tile), or
    MULTI_CORE_HW which also maps to ReduceSingleCoreHwProgramFactory

compute_program_hash() includes:
  math_op, dim, output_mem_config, output_dtype, compute_kernel_config,
  sub_core_grids, negate, scaler_mode, program_factory.index(), input dtype,
  input memory_config, input padded_shape.

Scalar values (scaler / post_mul_scaler) are runtime args and are excluded from the
hash (#54180). override_runtime_arguments() re-applies them on every hit.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def run_reduce_op(device, op, shape, dim, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Run a reduce op on device and return (torch_result, ttnn_result)."""
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_a = torch.rand(shape, dtype=torch_dtype) + 0.1

    ttnn_ops = {ttnn.sum: torch.sum, ttnn.max: torch.amax, ttnn.min: torch.amin}
    torch_result = ttnn_ops[op](torch_a, dim=dim, keepdim=True)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    with device.cache_entries_counter.measure():
        tt_result = op(tt_a, dim=dim, keepdim=True, memory_config=memory_config)
    tt_result = ttnn.to_torch(tt_result)

    return torch_result, tt_result


# =============================================================================
# Cache reuse tests (fields correctly excluded from hash)
# =============================================================================


def test_reduce_cache_reuse_same_config(device, isolate_program_cache):
    """Same op, same shape, same dtype run twice -> 1 cache entry, different outputs."""
    shape = [1, 1, 64, 64]

    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    torch.manual_seed(42)
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 1
    assert not torch.equal(tt_out1, tt_out2)


# =============================================================================
# Cache miss tests (fields correctly included in hash)
# =============================================================================


def test_reduce_cache_miss_different_math_ops(device, isolate_program_cache):
    """Different reduce math ops (sum vs max) -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.max, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_dims(device, isolate_program_cache):
    """Different reduce dims (W vs H) -> different program factories -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    # dim=-1 (W): ReduceMultiCoreWProgramFactory
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    # dim=-2 (H): ReduceMultiCoreHProgramFactory
    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-2, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_input_dtypes(device, isolate_program_cache):
    """Different input dtypes -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, shape, dim=-1, dtype=ttnn.float32)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.999,
        rtol=0.004,
        atol=0.152,
        frobenius_threshold=0.003,
    )
    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_memory_configs(device, isolate_program_cache):
    """Different memory configs -> different cache entries."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]

    torch_ref1, tt_out1 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    torch_ref2, tt_out2 = run_reduce_op(
        device, ttnn.sum, shape, dim=-1, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )

    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_different_shapes(device, isolate_program_cache):
    """Different padded shapes -> different cache entries.
    padded_shape is included in compute_program_hash() because Ht, Wt are compile-time args."""
    torch.manual_seed(0)
    torch_ref1, tt_out1 = run_reduce_op(device, ttnn.sum, [1, 1, 32, 64], dim=-1, dtype=ttnn.bfloat16)

    torch_ref2, tt_out2 = run_reduce_op(device, ttnn.sum, [1, 1, 64, 64], dim=-1, dtype=ttnn.bfloat16)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref1,
        tt_out1,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    # test for equivalance
    assert_numeric_metrics(
        torch_ref2,
        tt_out2,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )
    assert device.cache_entries_counter.total == 2


def test_reduce_cache_miss_sub_core_grids(device, isolate_program_cache):
    """Different sub_core_grids -> different cache entries.
    sub_core_grids is in compute_program_hash() and affects work distribution (compile-time)."""
    torch.manual_seed(0)
    shape = [1, 1, 64, 64]
    torch_a = torch.rand(shape, dtype=torch.bfloat16) + 0.1

    grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))])

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    with device.cache_entries_counter.measure():
        tt_out1 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_a)
        tt_out2 = ttnn.sum(tt_a, dim=-1, keepdim=True, sub_core_grids=grid_b)

    torch_ref = torch.sum(torch_a, dim=-1, keepdim=True)
    # test for equivalance
    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_out1),
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
    )
    # test for equivalence
    assert_numeric_metrics(
        torch_ref,
        ttnn.to_torch(tt_out2),
        pcc_threshold=0.999,
        rtol=0.007,
        atol=0.25,
        frobenius_threshold=0.001,
    )

    assert device.cache_entries_counter.total == 2


# Distinct scalars must reuse one program (#54180). Negative scalars are a separate case
# because max(s*x) with s<0 lowers to s*min(x), which is a different program.
TORCH_REDUCE = {
    ttnn.sum: lambda x, dim: torch.sum(x, dim=dim, keepdim=True),
    ttnn.max: lambda x, dim: torch.amax(x, dim=dim, keepdim=True),
    ttnn.min: lambda x, dim: torch.amin(x, dim=dim, keepdim=True),
}


@pytest.mark.parametrize("op", [ttnn.sum, ttnn.max, ttnn.min], ids=["sum", "max", "min"])
@pytest.mark.parametrize("dim", [-1, -2, [-2, -1]], ids=["dim_w", "dim_h", "dim_hw"])
@pytest.mark.parametrize("scalars", [(1.0, 2.0, 3.5, 0.25, 7.0), (-1.0, -2.0, -0.5)], ids=["pos", "neg"])
def test_reduce_scalar_value_does_not_add_cache_entries(device, isolate_program_cache, op, dim, scalars):
    """Several distinct scalars on one op/shape must not grow the program cache."""
    torch.manual_seed(0)
    # Ramp both axes so max/min outputs actually change; uniform noise is near-constant after reduce.
    row_ramp = 1.0 + torch.arange(64, dtype=torch.float32).reshape(1, 1, 64, 1) * 0.1
    col_ramp = 1.0 + torch.arange(64, dtype=torch.float32).reshape(1, 1, 1, 64) * 0.05
    torch_a = ((torch.rand([1, 1, 64, 64], dtype=torch.float32) + 0.1) * row_ramp * col_ramp).bfloat16()
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)

    entries = []
    for scalar in scalars:
        tt_out = op(tt_a, dim=dim, keepdim=True, scalar=scalar)
        assert_numeric_metrics(
            TORCH_REDUCE[op](scalar * torch_a.float(), dim),
            ttnn.to_torch(tt_out).float(),
            pcc_threshold=0.999,
            rtol=1e-02,
            atol=1e-02,
            frobenius_threshold=1e-01,
        )
        entries.append(device.num_program_cache_entries())

    assert len(set(entries)) == 1, (
        f"cache grew with the scalar value: entries={entries} for scalars={scalars}. "
        "The scalar must be a runtime arg, not part of the program hash."
    )


@pytest.mark.parametrize("op", [ttnn.sum, ttnn.max, ttnn.min], ids=["sum", "max", "min"])
@pytest.mark.parametrize("scalars", [(1.0, 2.0, 0.25), (-1.0, -2.0, -0.5)], ids=["pos", "neg"])
def test_single_core_hw_scalar_value_does_not_add_cache_entries(device, isolate_program_cache, op, scalars):
    """1-tile HW: identity scaler tile + runtime post-mul. Sign used to pick W-then-H."""
    torch.manual_seed(0)
    row_ramp = 1.0 + torch.arange(32, dtype=torch.float32).reshape(1, 1, 32, 1) * 0.1
    col_ramp = 1.0 + torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32) * 0.05
    torch_a = ((torch.rand([1, 1, 32, 32], dtype=torch.float32) + 0.1) * row_ramp * col_ramp).bfloat16()
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)

    entries = []
    for scalar in scalars:
        tt_out = op(tt_a, dim=[-2, -1], keepdim=True, scalar=scalar)
        assert_numeric_metrics(
            TORCH_REDUCE[op](scalar * torch_a.float(), [-2, -1]),
            ttnn.to_torch(tt_out).float(),
            pcc_threshold=0.999,
            rtol=1e-02,
            atol=1e-02,
            frobenius_threshold=1e-01,
        )
        entries.append(device.num_program_cache_entries())

    assert len(set(entries)) == 1, f"1-tile HW cache grew with the scalar: entries={entries} for scalars={scalars}."


def test_single_core_hw_sum_pos_and_neg_scalar_share_cache(device, isolate_program_cache):
    """SUM no longer forks topology on scaler sign (the old sqrt-NaN two-step)."""
    torch.manual_seed(0)
    torch_a = torch.rand([1, 1, 32, 32], dtype=torch.float32).bfloat16()
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn.sum(tt_a, dim=[-2, -1], keepdim=True, scalar=2.0)
    after_pos = device.num_program_cache_entries()
    ttnn.sum(tt_a, dim=[-2, -1], keepdim=True, scalar=-2.0)
    after_neg = device.num_program_cache_entries()

    assert (
        after_pos == after_neg
    ), f"1-tile HW SUM compiled a new program for the opposite sign: {after_pos} -> {after_neg}."
