"""
Tile-Based CUDA MIP Splatting
==============================

Wraps ``splat_mip_tiled_cuda`` — a tile-based rasterizer that assigns each
Gaussian to the screen tiles it overlaps, then launches one CUDA block per
tile, evaluating only that tile's Gaussians via shared-memory batching.

Pipeline
--------
1. **Python preprocessing** (on GPU, vectorised):
   compute 2D bounding box per Gaussian → tile assignments → sort by tile →
   build per-tile offset table.

2. **CUDA forward**: one thread block (16×16 = 256) per tile, shared-memory
   batch-loads Gaussian data, online softmax accumulation.

3. **CUDA backward**: same tile structure, ``atomicAdd`` per-Gaussian grads.

Complexity: O(K_tile) per pixel instead of O(K) (K=50k total, K_tile ≈ 50-200).

Public API
----------
* ``HAS_TILED_MIP_CUDA``  — True if the extension is importable.
* ``splat_mip_grid_tiled_cuda(H, W, means_2d, cov_2d, intensities, beta)``
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function

try:
    import splat_mip_tiled_cuda as _TC
    HAS_TILED_MIP_CUDA = True
except ImportError:
    HAS_TILED_MIP_CUDA = False

TILE_SIZE = 16
MAHAL_CUTOFF = 16.0


# ── helpers ──────────────────────────────────────────────────────────

def _invert_cov_2x2_packed(cov_2d: torch.Tensor) -> torch.Tensor:
    """Batch-invert (K, 2, 2) symmetric covariance → packed (K, 3)."""
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    det = (a * d - b * b).clamp(min=1e-12)
    inv_det = 1.0 / det
    return torch.stack([d * inv_det, -b * inv_det, a * inv_det], dim=-1)


# ── tile preprocessing (fully vectorised on GPU) ────────────────────

@torch.no_grad()
def build_tile_data(
    means_2d: torch.Tensor,   # (K, 2)
    cov_2d:   torch.Tensor,   # (K, 2, 2)
    H: int,
    W: int,
):
    """
    Compute per-tile Gaussian assignment data.

    Returns
    -------
    sorted_gauss_ids : (P,) int32  —  Gaussian indices sorted by tile
    tile_offsets     : (T+1,) int32 — cumsum; tile t owns [offsets[t], offsets[t+1])
    n_tiles_x, n_tiles_y : int
    """
    K = means_2d.shape[0]
    device = means_2d.device

    n_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    n_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    n_tiles   = n_tiles_x * n_tiles_y

    empty_ids     = torch.zeros(0, device=device, dtype=torch.int32)
    empty_offsets = torch.zeros(n_tiles + 1, device=device, dtype=torch.int32)

    if K == 0:
        return empty_ids, empty_offsets, n_tiles_x, n_tiles_y

    # 1. Axis-aligned bounding radius from covariance diagonal
    #    For mahal ≤ MAHAL_CUTOFF: extent_x = sqrt(cutoff · cov_xx)
    cov_xx = cov_2d[:, 0, 0].clamp(min=1e-8)
    cov_yy = cov_2d[:, 1, 1].clamp(min=1e-8)
    radius_x = torch.sqrt(MAHAL_CUTOFF * cov_xx)
    radius_y = torch.sqrt(MAHAL_CUTOFF * cov_yy)

    cx, cy = means_2d[:, 0], means_2d[:, 1]

    # 2. Tile index ranges (inclusive)
    tile_x_min = ((cx - radius_x).clamp(min=0.0) / TILE_SIZE).int().clamp(0, n_tiles_x - 1)
    tile_x_max = ((cx + radius_x).clamp(min=0.0, max=float(W) - 0.001) / TILE_SIZE).int().clamp(0, n_tiles_x - 1)
    tile_y_min = ((cy - radius_y).clamp(min=0.0) / TILE_SIZE).int().clamp(0, n_tiles_y - 1)
    tile_y_max = ((cy + radius_y).clamp(min=0.0, max=float(H) - 0.001) / TILE_SIZE).int().clamp(0, n_tiles_y - 1)

    tiles_per_x = (tile_x_max - tile_x_min + 1).clamp(min=0)
    tiles_per_y = (tile_y_max - tile_y_min + 1).clamp(min=0)
    tiles_per_gauss = (tiles_per_x * tiles_per_y).int()    # (K,)

    total_pairs = int(tiles_per_gauss.sum().item())
    if total_pairs == 0:
        return empty_ids, empty_offsets, n_tiles_x, n_tiles_y

    # 3. Expand Gaussian indices by tile count
    gauss_ids = torch.repeat_interleave(
        torch.arange(K, device=device, dtype=torch.int32),
        tiles_per_gauss,
    )                                                       # (P,)

    # Per-Gaussian offsets into the expanded array
    cumcounts = tiles_per_gauss.long().cumsum(0)
    offsets   = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        cumcounts[:-1],
    ])

    local_idx = torch.arange(total_pairs, device=device, dtype=torch.long) \
                - offsets[gauss_ids.long()]

    # Convert local index → (tile_dx, tile_dy)
    tpx     = tiles_per_x[gauss_ids.long()].long()
    tile_dx = (local_idx % tpx).int()
    tile_dy = (local_idx // tpx).int()

    tile_x = tile_x_min[gauss_ids.long()] + tile_dx
    tile_y = tile_y_min[gauss_ids.long()] + tile_dy

    tile_ids = (tile_y * n_tiles_x + tile_x).long()         # (P,)

    # 4. Sort by tile_id
    sorted_order   = tile_ids.argsort()
    sorted_tile_ids = tile_ids[sorted_order]
    sorted_gauss_ids = gauss_ids[sorted_order].contiguous()  # (P,) int32

    # 5. Per-tile cumulative offsets
    counts = torch.bincount(sorted_tile_ids, minlength=n_tiles)  # (T,) long
    tile_offsets = torch.zeros(n_tiles + 1, device=device, dtype=torch.int32)
    tile_offsets[1:] = counts.cumsum(0).to(torch.int32)

    return sorted_gauss_ids, tile_offsets, n_tiles_x, n_tiles_y


# ── autograd Function ───────────────────────────────────────────────

class _TiledSplatMIPFn(Function):
    """
    Forward:  (means_2d, cov_inv, intensities, sorted_ids, offsets, ...) → rendered
    Backward: grad_rendered → (grad_means_2d, grad_cov_inv, grad_intensities, ...)
    """

    @staticmethod
    def forward(ctx, means_2d, cov_inv, intensities,
                sorted_gauss_ids, tile_offsets,
                H, W, n_tiles_x, n_tiles_y, beta):
        rendered, max_bg, sum_exp = _TC.forward(
            means_2d, cov_inv, intensities,
            sorted_gauss_ids, tile_offsets,
            H, W, n_tiles_x, n_tiles_y,
            float(beta),
        )
        ctx.save_for_backward(
            means_2d, cov_inv, intensities,
            rendered, max_bg, sum_exp,
            sorted_gauss_ids, tile_offsets,
        )
        ctx.H, ctx.W = H, W
        ctx.n_tiles_x = n_tiles_x
        ctx.n_tiles_y = n_tiles_y
        ctx.beta = float(beta)
        return rendered

    @staticmethod
    def backward(ctx, grad_rendered):
        (means_2d, cov_inv, intensities,
         rendered, max_bg, sum_exp,
         sorted_gauss_ids, tile_offsets) = ctx.saved_tensors

        grad_means_2d, grad_cov_inv, grad_intensities = _TC.backward(
            grad_rendered.contiguous(),
            means_2d, cov_inv, intensities,
            sorted_gauss_ids, tile_offsets,
            rendered, max_bg, sum_exp,
            ctx.H, ctx.W, ctx.n_tiles_x, ctx.n_tiles_y,
            ctx.beta,
        )
        #   inputs: means_2d, cov_inv, intensities,
        #           sorted_gauss_ids, tile_offsets,
        #           H, W, n_tiles_x, n_tiles_y, beta
        return (grad_means_2d, grad_cov_inv, grad_intensities,
                None, None, None, None, None, None, None)


# ── public API ──────────────────────────────────────────────────────

def splat_mip_grid_tiled_cuda(
    H:           int,
    W:           int,
    means_2d:    torch.Tensor,
    cov_2d:      torch.Tensor,
    intensities: torch.Tensor,
    beta:        float = 50.0,
) -> torch.Tensor:
    """
    Tile-based CUDA MIP splatting — drop-in replacement for
    ``splat_mip_grid_cuda`` with O(K_tile) per pixel instead of O(K).

    Parameters
    ----------
    H, W         : image dims
    means_2d     : (K, 2)  projected Gaussian centres
    cov_2d       : (K, 2, 2) projected covariance matrices
    intensities  : (K,) signal amplitudes
    beta         : softmax temperature

    Returns
    -------
    rendered : (H*W,) flat
    """
    # Invert covariance (autograd traces this → grad flows to cov_2d)
    cov_inv = _invert_cov_2x2_packed(cov_2d)

    # Tile preprocessing (no grad)
    sorted_gauss_ids, tile_offsets, n_tiles_x, n_tiles_y = \
        build_tile_data(means_2d.detach(), cov_2d.detach(), H, W)

    return _TiledSplatMIPFn.apply(
        means_2d, cov_inv, intensities,
        sorted_gauss_ids, tile_offsets,
        H, W, n_tiles_x, n_tiles_y, beta,
    )
