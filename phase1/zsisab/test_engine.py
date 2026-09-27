"""Self-check for the coreset anchor refinement (Track A). Run directly: python zsisab/test_engine.py"""
from __future__ import annotations
import torch
from zsisab.engine import _refine_inducing_points


def _distortion(rows: torch.Tensor, centroids: torch.Tensor) -> float:
    dist = torch.cdist(rows, centroids) ** 2
    return dist.min(dim=-1).values.mean().item()


def test_refine_iters_zero_is_identity():
    torch.manual_seed(0)
    rows = torch.randn(1, 500, 8)
    idx = torch.randperm(500)[:32]
    out = _refine_inducing_points(rows, idx, M=32, chunk_size=128, iters=0)
    assert torch.equal(out, rows[:, idx, :])


def test_refine_reduces_distortion():
    torch.manual_seed(0)
    # 8 well-separated clusters so refinement has an obvious right answer.
    centers = torch.randn(8, 1, 16) * 10
    rows = (centers + torch.randn(8, 200, 16) * 0.5).view(1, -1, 16)
    idx = torch.randperm(rows.shape[1])[:8]

    random_anchors = rows[:, idx, :]
    refined = _refine_inducing_points(rows, idx, M=8, chunk_size=64, iters=5)

    before = _distortion(rows[0], random_anchors[0])
    after = _distortion(rows[0], refined[0])
    assert after < before, f"expected refinement to reduce distortion, got {after} >= {before}"


def test_shape_and_no_nans():
    torch.manual_seed(1)
    rows = torch.randn(2, 1000, 12)
    idx = torch.randperm(1000)[:64]
    out = _refine_inducing_points(rows, idx, M=64, chunk_size=300, iters=3)
    assert out.shape == (2, 64, 12)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_refine_iters_zero_is_identity()
    test_refine_reduces_distortion()
    test_shape_and_no_nans()
    print("All zsisab/test_engine.py checks passed.")
