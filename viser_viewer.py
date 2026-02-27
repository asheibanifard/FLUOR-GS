#!/usr/bin/env python3
"""
viser_viewer.py — Interactive real-time MIP Gaussian Viewer
============================================================

Renders MIP projections on the fly using the tile-based CUDA kernel.
Each camera update triggers a fresh render (~7-11 ms at 512×512).

Usage:
    cd /workspace/hisnegs/src/renderer
    python viser_viewer.py
    python viser_viewer.py --ckpt ../checkpoints/mip_ckpt/e2e_ep400.pt
    python viser_viewer.py --port 8080 --res 512
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import viser

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rendering import (
    Camera,
    GaussianParameters,
    compute_aspect_scales,
    apply_aspect_correction,
    render_mip_projection,
    load_config,
)


# =====================================================================
#  Load & build Gaussian parameters on GPU
# =====================================================================
def load_gaussians(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded {ckpt_path}")
    print(f"  Epoch {ckpt['epoch']},  K = {ckpt['means'].shape[0]}")

    means           = ckpt["means"].float().to(device)
    log_scales      = ckpt["log_scales"].float().to(device)
    quaternions     = ckpt["quaternions"].float().to(device)
    log_intensities = ckpt["log_intensities"].float().to(device)
    K = means.shape[0]

    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(K, 3, 3, device=device)
    R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-w*z); R[:, 0, 2] = 2*(x*z+w*y)
    R[:, 1, 0] = 2*(x*y+w*z);   R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-w*x)
    R[:, 2, 0] = 2*(x*z-w*y);   R[:, 2, 1] = 2*(y*z+w*x);   R[:, 2, 2] = 1-2*(x*x+y*y)
    S2  = torch.diag_embed(scales ** 2)
    cov = R @ S2 @ R.transpose(-2, -1)
    intensities = torch.sigmoid(log_intensities)
    return means, cov, intensities, ckpt["epoch"]


# =====================================================================
#  Extract R, T from viser camera
# =====================================================================
def camera_to_RT(cam_handle, device: torch.device):
    """Convert viser camera (position, look_at, up) → (3×3 R, 3 T)."""
    pos    = np.asarray(cam_handle.position, dtype=np.float64)
    look   = np.asarray(cam_handle.look_at,  dtype=np.float64)
    up_dir = np.asarray(cam_handle.up_direction, dtype=np.float64)

    forward = look - pos
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward /= fn

    right = np.cross(forward, up_dir)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn

    up = np.cross(right, forward)

    # Match _orbit_pose convention: X=right, Y=-up, Z=forward
    # Z points toward the scene → positive z for visible objects
    R_np = np.stack([right, -up, forward], axis=0)  # (3, 3)
    T_np = R_np @ (-pos)

    R_t = torch.from_numpy(R_np).float().to(device)
    T_t = torch.from_numpy(T_np).float().to(device)
    return R_t, T_t


# =====================================================================
#  Render one MIP frame → (H, W, 3) uint8
# =====================================================================
@torch.no_grad()
def render_mip_frame(gaussians, camera, R_cam, T_cam, beta, device):
    img, n_vis = render_mip_projection(
        gaussians, camera, R_cam, T_cam,
        beta=beta, chunk_size=4096,
    )
    img_np = img.cpu().numpy()
    vmax = img_np.max() + 1e-8
    gray = np.clip(img_np / vmax * 255, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1), n_vis


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive MIP Gaussian Viewer (viser)")
    parser.add_argument("--ckpt", default="../checkpoints/mip_ckpt/e2e_ep800.pt")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--res", type=int, default=256, help="MIP render resolution")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.ckpt))

    # ── Load model ──
    means, cov, intensities, epoch = load_gaussians(ckpt_path, device)
    K = means.shape[0]

    # ── Aspect correction ──
    vol_shape = (100, 647, 813)
    aspect_scales = compute_aspect_scales(vol_shape).to(device)
    gaussians = GaussianParameters(means=means, covariances=cov, intensities=intensities)
    gaussians = apply_aspect_correction(gaussians, aspect_scales)

    # ── Camera matching training config ──
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config_splat.yml"))
    res = args.res
    cam_render = Camera.from_config(cfg, width=res, height=res)
    default_radius = cfg["poses"]["radius"]
    default_beta   = cfg["training"]["beta_mip"]

    # ── Warm-up render ──
    from rendering import _orbit_pose
    R0, T0 = _orbit_pose(15.0, 0.0, default_radius)
    R0, T0 = R0.to(device), T0.to(device)
    init_frame, _ = render_mip_frame(gaussians, cam_render, R0, T0, default_beta, device)
    torch.cuda.synchronize()
    print("  Warm-up render complete")

    # ── Viser server ──
    server = viser.ViserServer(port=args.port)
    server.initial_camera.position     = (0.0, 0.0, float(default_radius))
    server.initial_camera.look_at      = (0.0, 0.0, 0.0)
    server.initial_camera.fov          = math.radians(cfg["camera"]["fov_x_deg"])
    server.initial_camera.up_direction = (0.0, 1.0, 0.0)

    # ── GUI ──
    with server.gui.add_folder("MIP Settings"):
        beta_slider = server.gui.add_slider(
            "Beta (MIP sharpness)", min=1.0, max=100.0, step=1.0,
            initial_value=default_beta,
        )

    status = server.gui.add_markdown(
        f"**Epoch {epoch}** | K={K} | {res}x{res} | Real-time tiled MIP | Drag to rotate"
    )

    # ── Serve initial image ──
    server.scene.set_background_image(init_frame, format="jpeg", jpeg_quality=90)

    # ── Per-client: render live on every camera move ──
    def render_and_serve(client, cam_handle):
        t0 = time.time()
        R_cam, T_cam = camera_to_RT(cam_handle, device)
        frame, n_vis = render_mip_frame(
            gaussians, cam_render, R_cam, T_cam, beta_slider.value, device,
        )
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        client.scene.set_background_image(frame, format="jpeg", jpeg_quality=85)
        status.content = (
            f"**Epoch {epoch}** | K={K} | {res}x{res} | "
            f"{n_vis} visible | {ms:.0f} ms ({1000/max(ms,1):.0f} FPS)"
        )

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        print(f"  Client {client.client_id} connected")

        @client.camera.on_update
        def _(cam_handle) -> None:
            render_and_serve(client, cam_handle)

        render_and_serve(client, client.camera)

    print(f"\n--- Interactive MIP Gaussian Viewer (real-time) ---")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  {res}x{res} MIP, tiled CUDA kernel")
    print(f"  Epoch {epoch}, K={K}, β={default_beta}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()


if __name__ == "__main__":
    main()
