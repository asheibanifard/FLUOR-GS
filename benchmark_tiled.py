#!/usr/bin/env python3
"""
Benchmark: tiled vs non-tiled MIP splatting CUDA kernels
=========================================================
Loads a checkpoint, renders at multiple resolutions, compares:
  1. Correctness: max absolute difference between tiled and flat outputs
  2. Speed: forward pass timing (ms) for both kernels
  3. Tile statistics: average Gaussians per tile
"""

import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/hisnegs/src/renderer')

from rendering import (
    GaussianParameters, Camera, load_config, compute_aspect_scales,
    apply_aspect_correction, _orbit_pose, transform_to_camera, project_to_2d,
)
from splat_mip_cuda_wrapper import splat_mip_grid_cuda, HAS_MIP_CUDA
from splat_mip_tiled_wrapper import (
    splat_mip_grid_tiled_cuda, HAS_TILED_MIP_CUDA, build_tile_data,
    _invert_cov_2x2_packed,
)

assert HAS_MIP_CUDA, "Need non-tiled CUDA kernel for comparison"
assert HAS_TILED_MIP_CUDA, "Need tiled CUDA kernel"

device = torch.device('cuda')

# ── load checkpoint ─────────────────────────────────────────────────
ckpt = torch.load('/workspace/hisnegs/src/checkpoints/mip_ckpt/e2e_ep400.pt',
                   map_location=device)
means = ckpt['means'].float().to(device)
log_scales = ckpt['log_scales'].float().to(device)
quaternions = ckpt['quaternions'].float().to(device)
log_intensities = ckpt['log_intensities'].float().to(device)
K = means.shape[0]
print(f"Checkpoint: K = {K} Gaussians")

# Build Gaussian parameters
scales = torch.exp(log_scales).clamp(1e-5, 1e2)
q = F.normalize(quaternions, p=2, dim=-1)
w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
R_rot = torch.zeros(K,3,3,device=device)
R_rot[:,0,0]=1-2*(y*y+z*z); R_rot[:,0,1]=2*(x*y-w*z); R_rot[:,0,2]=2*(x*z+w*y)
R_rot[:,1,0]=2*(x*y+w*z); R_rot[:,1,1]=1-2*(x*x+z*z); R_rot[:,1,2]=2*(y*z-w*x)
R_rot[:,2,0]=2*(x*z-w*y); R_rot[:,2,1]=2*(y*z+w*x); R_rot[:,2,2]=1-2*(x*x+y*y)
S2 = torch.diag_embed(scales**2)
cov = R_rot @ S2 @ R_rot.transpose(-2,-1)
intensities = torch.sigmoid(log_intensities)

vol_shape = (100, 647, 813)
aspect_scales = compute_aspect_scales(vol_shape).to(device)
gaussians = GaussianParameters(means=means, covariances=cov, intensities=intensities)
gaussians = apply_aspect_correction(gaussians, aspect_scales)

cfg = load_config('/workspace/hisnegs/src/renderer/config_splat.yml')
radius = cfg['poses']['radius']
beta = cfg['training']['beta_mip']

# ── camera setup ────────────────────────────────────────────────────
R_c, T_c = _orbit_pose(15.0, 45.0, radius)
R_c, T_c = R_c.to(device), T_c.to(device)


def prepare_2d(cam):
    """Transform 3D Gaussians to 2D for the camera."""
    means_cam, cov_cam = transform_to_camera(gaussians.means, gaussians.covariances, R_c, T_c)
    z = means_cam[:, 2]
    vis = (z > cam.near) & (z < cam.far)
    means_cam_v = means_cam[vis]
    cov_cam_v = cov_cam[vis]
    intens_v = gaussians.intensities[vis]
    means_2d, cov_2d, _ = project_to_2d(means_cam_v, cov_cam_v, cam)
    # Cull off-screen
    with torch.no_grad():
        a_c, b_c, d_c = cov_2d[:, 0, 0], cov_2d[:, 0, 1], cov_2d[:, 1, 1]
        tr = a_c + d_c
        disc = (tr * tr - 4.0 * (a_c * d_c - b_c * b_c)).clamp(min=0.0)
        rad = 3.0 * torch.sqrt(0.5 * (tr + torch.sqrt(disc)).clamp(min=1e-8))
        u, v = means_2d[:, 0], means_2d[:, 1]
        in_img = ((u + rad > 0) & (u - rad < cam.width) &
                  (v + rad > 0) & (v - rad < cam.height))
    return means_2d[in_img], cov_2d[in_img], intens_v[in_img]


print("\n" + "="*70)
print("TILE-BASED vs FLAT MIP SPLATTING BENCHMARK")
print("="*70)

for res in [128, 256, 512, 1024]:
    cam = Camera.from_config(cfg, width=res, height=res)
    means_2d, cov_2d, intens = prepare_2d(cam)
    K_vis = means_2d.shape[0]
    H, W = cam.height, cam.width

    # ── correctness check ───────────────────────────────────────
    with torch.no_grad():
        out_flat  = splat_mip_grid_cuda(H, W, means_2d, cov_2d, intens, beta)
        out_tiled = splat_mip_grid_tiled_cuda(H, W, means_2d, cov_2d, intens, beta)
    max_diff = (out_flat - out_tiled).abs().max().item()
    mean_diff = (out_flat - out_tiled).abs().mean().item()

    # ── tile statistics ─────────────────────────────────────────
    with torch.no_grad():
        _, tile_offsets, ntx, nty = build_tile_data(means_2d, cov_2d, H, W)
        tile_counts = tile_offsets[1:] - tile_offsets[:-1]
        avg_per_tile = tile_counts.float().mean().item()
        max_per_tile = tile_counts.max().item()
        total_pairs = tile_offsets[-1].item()

    # ── timing ──────────────────────────────────────────────────
    N_ITERS = 50

    # Warm up
    for _ in range(5):
        splat_mip_grid_cuda(H, W, means_2d, cov_2d, intens, beta)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITERS):
        splat_mip_grid_cuda(H, W, means_2d, cov_2d, intens, beta)
    torch.cuda.synchronize()
    ms_flat = (time.time() - t0) / N_ITERS * 1000

    # Warm up tiled
    for _ in range(5):
        splat_mip_grid_tiled_cuda(H, W, means_2d, cov_2d, intens, beta)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITERS):
        splat_mip_grid_tiled_cuda(H, W, means_2d, cov_2d, intens, beta)
    torch.cuda.synchronize()
    ms_tiled = (time.time() - t0) / N_ITERS * 1000

    speedup = ms_flat / ms_tiled if ms_tiled > 0 else float('inf')

    print(f"\n--- {res}×{res} ({K_vis} visible Gaussians) ---")
    print(f"  Tiles: {ntx}×{nty} = {ntx*nty} tiles")
    print(f"  Gauss/tile:  avg={avg_per_tile:.0f}  max={max_per_tile}  total_pairs={total_pairs:,}")
    print(f"  Correctness: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    print(f"  Flat kernel:  {ms_flat:.2f} ms  ({1000/ms_flat:.1f} FPS)")
    print(f"  Tiled kernel: {ms_tiled:.2f} ms  ({1000/ms_tiled:.1f} FPS)")
    print(f"  Speedup:      {speedup:.1f}×")

# ── gradient check ──────────────────────────────────────────────────
print("\n" + "="*70)
print("GRADIENT CHECK (256×256)")
print("="*70)
cam = Camera.from_config(cfg, width=256, height=256)
means_2d, cov_2d, intens = prepare_2d(cam)
means_2d = means_2d.clone().requires_grad_(True)
cov_2d = cov_2d.clone().requires_grad_(True)
intens = intens.clone().requires_grad_(True)

out_tiled = splat_mip_grid_tiled_cuda(256, 256, means_2d, cov_2d, intens, beta)
loss = out_tiled.sum()
loss.backward()
print(f"  grad_means_2d:  min={means_2d.grad.min().item():.4e}  max={means_2d.grad.max().item():.4e}  norm={means_2d.grad.norm().item():.4e}")
print(f"  grad_cov_2d:    min={cov_2d.grad.min().item():.4e}  max={cov_2d.grad.max().item():.4e}  norm={cov_2d.grad.norm().item():.4e}")
print(f"  grad_intens:    min={intens.grad.min().item():.4e}  max={intens.grad.max().item():.4e}  norm={intens.grad.norm().item():.4e}")
print("  ✓ Backward pass completed without error")

print("\nDone.")
