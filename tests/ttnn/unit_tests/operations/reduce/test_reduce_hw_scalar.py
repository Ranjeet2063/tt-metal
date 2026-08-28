# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HW reduce with a negative scalar.

1-tile HW uses an identity scaler tile and a runtime post-mul, so a negative
scalar stays on ReduceSingleCoreHwProgramFactory. Multi-tile HW is W-then-H.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_numeric_metrics, assert_with_ulp

pytestmark = pytest.mark.use_module_device

SCALAR = -2.5

OPS = [
    (ttnn.sum, lambda x, dim: torch.sum(x, dim=dim, keepdim=True)),
    (ttnn.max, lambda x, dim: torch.amax(x, dim=dim, keepdim=True)),
    (ttnn.min, lambda x, dim: torch.amin(x, dim=dim, keepdim=True)),
]


def _compare(torch_ref, tt_out, dtype, fast_and_approximate_mode, device, op):
    if dtype == ttnn.int32:
        assert_equal(torch_ref, tt_out)
        return
    if dtype == ttnn.float32 and not fast_and_approximate_mode and device.arch() != ttnn.device.Arch.QUASAR:
        if op is ttnn.sum:
            # 4096-element HW sum is not bit-exact even on the SFPU path.
            assert_with_ulp(torch_ref, tt_out, ulp_threshold=2)
        else:
            assert_equal(torch_ref, tt_out)
        return
    if dtype == ttnn.float32:
        # Fast FPU/TF32: 64x64 sum of randn * -2.5 misses by ~0.25 ATOL.
        assert_allclose(torch_ref, tt_out, rtol=1e-2, atol=0.3)
        return
    assert_numeric_metrics(
        torch_ref,
        tt_out.float(),
        pcc_threshold=0.999,
        rtol=1e-02,
        atol=1e-02,
        frobenius_threshold=1e-01,
    )


def _run_hw_negative_scalar(device, op, torch_reduce, shape, dim, dtype, fast_and_approximate_mode):
    torch.manual_seed(0)
    if dtype == ttnn.int32:
        torch_a = torch.randint(-50, 50, shape, dtype=torch.int32)
        torch_ref = torch_reduce(torch_a.float() * SCALAR, dim).to(torch.int32)
    else:
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        torch_a = torch.randn(shape, dtype=torch_dtype)
        torch_ref = torch_reduce(SCALAR * torch_a.float(), dim)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    op_kwargs = {}
    if dtype == ttnn.float32:
        op_kwargs["fast_and_approximate_mode"] = fast_and_approximate_mode
    tt_out = ttnn.to_torch(op(tt_a, dim=dim, keepdim=True, scalar=SCALAR, **op_kwargs))
    _compare(torch_ref, tt_out.reshape(torch_ref.shape), dtype, fast_and_approximate_mode, device, op)


@pytest.mark.parametrize("op, torch_reduce", OPS, ids=["sum", "max", "min"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 64, 64)], ids=["single_tile", "multi_tile"])
@pytest.mark.parametrize(
    "dtype, fast_and_approximate_mode",
    [
        (ttnn.bfloat16, True),
        (ttnn.float32, True),
        (ttnn.float32, False),
        (ttnn.int32, True),
    ],
    ids=["bf16", "fp32_fast", "fp32_accurate", "int32"],
)
def test_hw_negative_scalar(device, op, torch_reduce, shape, dtype, fast_and_approximate_mode):
    """dim=HW, scalar<0: 1-tile uses the single-core factory; multi-tile stays W-then-H."""
    _run_hw_negative_scalar(device, op, torch_reduce, shape, [-2, -1], dtype, fast_and_approximate_mode)


@pytest.mark.parametrize("shape, dim", [((2, 3, 32, 32), (0, -2, -1)), ((2, 3, 16, 24), (-3, -1))])
@pytest.mark.parametrize(
    "dtype, fast_and_approximate_mode",
    [
        (ttnn.bfloat16, True),
        (ttnn.float32, False),
        (ttnn.int32, True),
    ],
    ids=["bf16", "fp32_accurate", "int32"],
)
def test_min_multidim_negative_scalar(device, shape, dim, dtype, fast_and_approximate_mode):
    """nd-loop MIN applies the scalar only on the last axis; PostMul must commute with -MAX(-x)."""
    _run_hw_negative_scalar(
        device, ttnn.min, lambda x, d: torch.amin(x, dim=d, keepdim=True), shape, dim, dtype, fast_and_approximate_mode
    )
