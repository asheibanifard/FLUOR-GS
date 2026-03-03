#!/usr/bin/env python3
"""
view_ply.py — Interactive Gaussian Splat Viewer (rendering.py MIP + viser, HTTPS)

Uses the NeuroSGM MIP splatting pipeline from rendering.py instead of gsplat.
Each RGB channel is rendered as a separate intensity through soft-max MIP.

Usage:
    python view_ply.py                                           # default PLY
    python view_ply.py --ply dataset/abdomen.ply
    python view_ply.py --ply checkpoints/gsplat_ckpt/gsplat_final.ply
    python view_ply.py --port 8080 --https-port 8443 --res 800
"""

import argparse
import asyncio
import math
import os
import socket
import ssl
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from plyfile import PlyData
from PIL import Image

import viser
import gsplat

# Import MIP rendering pipeline from rendering.py
from rendering import (
    Camera,
    GaussianParameters,
    render_mip_projection,
)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive Gaussian Splat PLY Viewer")
    p.add_argument("--ply", default="3DGS_PLY_sample_data/PLY(postshot)/cactus_splat3_30kSteps_719k_splats.ply",
                    help="Path to .ply file")
    p.add_argument("--max-splats", type=int, default=None,
                    help="Max splats to load (None = all)")
    p.add_argument("--res", type=int, default=800, help="Render resolution (square)")
    p.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    p.add_argument("--https-port", type=int, default=8443, help="HTTPS proxy port")
    p.add_argument("--no-https", action="store_true", help="Disable HTTPS proxy")
    p.add_argument("--fov", type=float, default=60.0, help="Vertical FOV in degrees")
    p.add_argument("--dist-mult", type=float, default=2.5,
                    help="Camera distance as multiple of scene radius")
    p.add_argument("--beta", type=float, default=50.0,
                    help="Soft-max sharpness for MIP splatting")
    p.add_argument("--save-samples", type=str, default=None, metavar="DIR",
                    help="Save sample renders to DIR and exit (no interactive viewer)")
    p.add_argument("--n-angles", type=int, default=8,
                    help="Number of azimuth angles for --save-samples")
    p.add_argument("--elevations", type=float, nargs="+", default=[0.0, 30.0, -30.0],
                    help="Elevation angles (degrees) for --save-samples")
    return p.parse_args()


def load_ply(ply_path, max_splats, device):
    print(f"Loading {ply_path} ...")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    props = [p.name for p in vertex.properties]
    print(f"  {vertex.count:,} vertices, {len(props)} properties")

    df = pd.DataFrame({name: vertex[name] for name in props})
    if max_splats and len(df) > max_splats:
        df = df.sample(max_splats, random_state=42)
        print(f"  Sub-sampled to {len(df):,} splats")

    N = len(df)
    means = torch.tensor(
        np.stack([df["x"].values, df["y"].values, df["z"].values], axis=-1),
        dtype=torch.float32, device=device,
    )
    log_scales = torch.tensor(
        np.stack([df["scale_0"].values, df["scale_1"].values, df["scale_2"].values], axis=-1),
        dtype=torch.float32, device=device,
    )
    scales = torch.exp(log_scales).clamp(1e-6, 10.0)

    quats = torch.tensor(
        np.stack([df["rot_0"].values, df["rot_1"].values,
                  df["rot_2"].values, df["rot_3"].values], axis=-1),
        dtype=torch.float32, device=device,
    )
    quats = F.normalize(quats, dim=-1)

    # Build full 3×3 covariance matrices: Cov = R @ diag(s²) @ Rᵀ
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    K = N
    Rm = torch.zeros(K, 3, 3, device=device, dtype=torch.float32)
    Rm[:, 0, 0] = 1 - 2*(y*y + z*z);  Rm[:, 0, 1] = 2*(x*y - w*z);  Rm[:, 0, 2] = 2*(x*z + w*y)
    Rm[:, 1, 0] = 2*(x*y + w*z);      Rm[:, 1, 1] = 1 - 2*(x*x+z*z);Rm[:, 1, 2] = 2*(y*z - w*x)
    Rm[:, 2, 0] = 2*(x*z - w*y);      Rm[:, 2, 1] = 2*(y*z + w*x);  Rm[:, 2, 2] = 1 - 2*(x*x+y*y)
    S2  = torch.diag_embed(scales ** 2)   # (K, 3, 3)
    covariances = Rm @ S2 @ Rm.transpose(-2, -1)  # (K, 3, 3)

    if all(c in df.columns for c in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        sh_dc = torch.tensor(
            np.stack([df["f_dc_0"].values, df["f_dc_1"].values, df["f_dc_2"].values], axis=-1),
            dtype=torch.float32, device=device,
        )
        colors_rgb = torch.sigmoid(sh_dc)  # (N, 3)
    else:
        colors_rgb = torch.full((N, 3), 0.5, device=device)

    # Per-channel intensities for separate MIP rendering
    intens_r = colors_rgb[:, 0]  # (N,)  red channel intensity
    intens_g = colors_rgb[:, 1]  # (N,)  green channel intensity
    intens_b = colors_rgb[:, 2]  # (N,)  blue channel intensity
    # Luminance as combined intensity
    intens_lum = 0.2126 * intens_r + 0.7152 * intens_g + 0.0722 * intens_b

    # Original 3DGS parameters for alpha-blending path
    opacities = torch.sigmoid(torch.tensor(df["opacity"].values, dtype=torch.float32, device=device)) if "opacity" in df.columns else torch.ones(N, device=device)

    print(f"  Loaded {N:,} splats onto {device}")
    return means, quats, scales, covariances, opacities, intens_r, intens_g, intens_b, intens_lum, colors_rgb, N


def setup_https_proxy(viser_port, https_port, cert_dir="/tmp/viser_ssl"):
    """TCP-level SSL proxy: terminates TLS then forwards raw bytes to viser.
    This handles HTTP, WebSocket upgrades, and all other traffic transparently."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        print("Generating self-signed SSL certificate ...")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_file, "-out", cert_file,
            "-days", "365", "-nodes", "-subj", "/CN=localhost",
        ], check=True, capture_output=True)
        print(f"  cert: {cert_file}")
        print(f"  key:  {key_file}")
    else:
        print(f"Reusing SSL cert from {cert_dir}")

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(cert_file, key_file)

    async def pipe(reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def handle_client(client_reader, client_writer):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                "127.0.0.1", viser_port
            )
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(
            pipe(client_reader, upstream_writer),
            pipe(upstream_reader, client_writer),
        )

    async def run_server():
        srv = await asyncio.start_server(
            handle_client, "0.0.0.0", https_port, ssl=ssl_ctx
        )
        async with srv:
            await srv.serve_forever()

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"HTTPS/WSS proxy on port {https_port} -> viser port {viser_port}")


def _orbit_pose(elevation_deg, azimuth_deg, radius, device):
    """Generate camera R, T from orbit parameters (elevation, azimuth, radius)."""
    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    cam_pos = torch.tensor([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=torch.float32)
    forward = -cam_pos / (cam_pos.norm() + 1e-8)

    up_y = torch.tensor([0.0, 1.0, 0.0])
    up_z = torch.tensor([0.0, 0.0, 1.0])
    pole_weight = forward[1].abs()
    world_up = (1.0 - pole_weight) * up_y + pole_weight * up_z
    world_up = world_up / (world_up.norm() + 1e-8)

    right = torch.linalg.cross(forward, world_up)
    right = right / (right.norm() + 1e-8)
    up = torch.linalg.cross(right, forward)
    up = up / (up.norm() + 1e-8)
    R = torch.stack([right, -up, forward], dim=0).to(device)
    T = (-R @ cam_pos.to(device))
    return R, T


def cam_to_RT(cam_handle, device):
    """Convert viser camera handle to R (3×3) and T (3,) for rendering.py."""
    pos = np.asarray(cam_handle.position, dtype=np.float64)
    look = np.asarray(cam_handle.look_at, dtype=np.float64)
    up_dir = np.asarray(cam_handle.up_direction, dtype=np.float64)

    fwd = look - pos
    fn = np.linalg.norm(fwd)
    fwd = fwd / fn if fn > 1e-8 else np.array([0.0, 0.0, -1.0])

    right = np.cross(fwd, up_dir)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-8 else np.array([1.0, 0.0, 0.0])

    up = np.cross(right, fwd)

    R = np.stack([right, -up, fwd], axis=0)     # (3, 3)
    T = R @ (-pos)                               # (3,)

    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    return R_t, T_t


def cam_to_viewmat(cam_handle, device):
    """Convert viser camera handle to 4×4 viewmat for gsplat."""
    R, T = cam_to_RT(cam_handle, device)
    vm = torch.eye(4, device=device, dtype=torch.float32)
    vm[:3, :3] = R
    vm[:3, 3] = T
    return vm


def main():
    args = parse_args()

    # Resolve PLY path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.ply):
        args.ply = os.path.join(script_dir, args.ply)

    # Kill any leftover processes on our ports
    for port in [args.port, args.https_port]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.bind(("0.0.0.0", port))
            s.close()
        except OSError:
            print(f"  Port {port} in use, attempting to free it ...")
            os.system(f"fuser -k {port}/tcp 2>/dev/null")
            time.sleep(0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    means, quats, scales, covariances, opacities, intens_r, intens_g, intens_b, intens_lum, colors_rgb, N = load_ply(
        args.ply, args.max_splats, device
    )

    IMG_H = IMG_W = args.res
    NEAR, FAR = 0.01, 100.0
    BETA = args.beta

    # Camera intrinsics via rendering.py Camera dataclass (for MIP)
    camera = Camera.from_fov(
        fov_x_deg=args.fov, width=IMG_W, height=IMG_H,
        near=NEAR, far=FAR,
    )

    # Intrinsics matrix for gsplat alpha-blending
    fov_y = math.radians(args.fov)  # for viser initial camera
    fy = IMG_H / (2.0 * math.tan(fov_y / 2.0))
    fx = fy
    K_intr = torch.tensor([[fx, 0, IMG_W / 2.0],
                            [0, fy, IMG_H / 2.0],
                            [0, 0, 1.0]], device=device)

    # Scene bounds
    centroid = means.mean(dim=0)
    scene_radius = (means - centroid).norm(dim=-1).quantile(0.95).item()
    cam_dist = scene_radius * args.dist_mult

    # Channel intensity maps (for MIP path)
    channel_map = {
        "R": intens_r,
        "G": intens_g,
        "B": intens_b,
        "Lum": intens_lum,
    }

    # ── MIP rendering functions ──────────────────────────────────
    @torch.no_grad()
    def render_channel_mip(R, T, intensities):
        """Render a single intensity channel via MIP splatting.
        Returns (H, W) tensor in [0, 1]."""
        gaussians = GaussianParameters(
            means=means, covariances=covariances, intensities=intensities,
        )
        img, n_vis = render_mip_projection(
            gaussians, camera, R, T, beta=BETA,
        )
        return img.clamp(0, 1)  # (H, W)

    @torch.no_grad()
    def render_mip_rgb(R, T):
        """MIP-composite RGB: render each channel separately via MIP, stack."""
        r = render_channel_mip(R, T, intens_r)
        g = render_channel_mip(R, T, intens_g)
        b = render_channel_mip(R, T, intens_b)
        return torch.stack([r, g, b], dim=-1)  # (H, W, 3)

    # ── Alpha-blending rendering (gsplat) ────────────────────────
    @torch.no_grad()
    def render_alpha_rgb(viewmat):
        """Standard 3DGS alpha-blending via gsplat rasterization."""
        renders, _, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors_rgb,
            viewmats=viewmat.unsqueeze(0), Ks=K_intr.unsqueeze(0),
            width=IMG_W, height=IMG_H,
            near_plane=NEAR, far_plane=FAR,
            render_mode="RGB", packed=True, rasterize_mode="antialiased",
        )
        return renders[0].clamp(0, 1)  # (H, W, 3)

    # ── Combined render_view with all modes ──────────────────────
    @torch.no_grad()
    def render_view(R, T, viewmat, mode="Compare"):
        """Render based on selected mode.
        Modes:
          'Compare'    → side-by-side: Alpha-Blend (left) | MIP (right)
          'Alpha-Blend'→ standard 3DGS alpha-compositing (gsplat)
          'MIP RGB'    → per-channel MIP composite
          'MIP R/G/B'  → single MIP channel greyscale
          'MIP All'    → 2×2 grid: R | G | B | RGB (MIP)
        """
        if mode == "Alpha-Blend":
            img = render_alpha_rgb(viewmat)  # (H, W, 3)
        elif mode == "MIP RGB":
            img = render_mip_rgb(R, T)
        elif mode in ("MIP R", "MIP G", "MIP B", "MIP Lum"):
            ch = mode.split()[-1]  # 'R', 'G', 'B', 'Lum'
            grey = render_channel_mip(R, T, channel_map[ch])  # (H, W)
            img = grey.unsqueeze(-1).expand(-1, -1, 3)
        elif mode == "MIP All":
            r = render_channel_mip(R, T, intens_r)
            g = render_channel_mip(R, T, intens_g)
            b = render_channel_mip(R, T, intens_b)
            rgb = torch.stack([r, g, b], dim=-1)
            z = torch.zeros_like(r)
            r_t = torch.stack([r, z, z], dim=-1)
            g_t = torch.stack([z, g, z], dim=-1)
            b_t = torch.stack([z, z, b], dim=-1)
            top = torch.cat([r_t, g_t], dim=1)
            bot = torch.cat([b_t, rgb], dim=1)
            img = torch.cat([top, bot], dim=0)
        else:  # "Compare" — side-by-side
            alpha_img = render_alpha_rgb(viewmat)            # (H, W, 3)
            mip_img   = render_mip_rgb(R, T)                # (H, W, 3)
            # Add thin white separator line
            sep = torch.ones(IMG_H, 2, 3, device=alpha_img.device)
            img = torch.cat([alpha_img, sep, mip_img], dim=1)  # (H, 2W+2, 3)
        return (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    # ── Save sample renders if requested ──────────────────────────
    if args.save_samples:
        save_dir = args.save_samples
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving sample renders to {save_dir}/")
        print(f"  angles: {args.n_angles} azimuths × {len(args.elevations)} elevations")
        print(f"  resolution: {IMG_W}×{IMG_H}, β={BETA}")

        view_idx = 0
        for el in args.elevations:
            for az_i in range(args.n_angles):
                az = 360.0 * az_i / args.n_angles
                R, T = _orbit_pose(el, az, cam_dist, device)
                # Translate camera to orbit around the centroid
                T = T + R @ centroid

                # Render each channel separately via MIP
                r_img = render_channel_mip(R, T, intens_r)   # (H, W)
                g_img = render_channel_mip(R, T, intens_g)   # (H, W)
                b_img = render_channel_mip(R, T, intens_b)   # (H, W)

                tag = f"el{el:+.0f}_az{az:03.0f}"

                # Save individual channels as greyscale PNGs
                for ch_name, ch_tensor in [("R", r_img), ("G", g_img), ("B", b_img)]:
                    arr = (ch_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    Image.fromarray(arr, mode="L").save(
                        os.path.join(save_dir, f"{view_idx:03d}_{tag}_{ch_name}.png")
                    )

                # Combine channels — MIP-composited RGB
                rgb = torch.stack([r_img, g_img, b_img], dim=-1).clamp(0, 1)
                rgb_arr = (rgb * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(rgb_arr, mode="RGB").save(
                    os.path.join(save_dir, f"{view_idx:03d}_{tag}_RGB.png")
                )

                # Also save the 2×2 grid (R-tinted | G-tinted | B-tinted | RGB)
                z = torch.zeros_like(r_img)
                r_t = torch.stack([r_img, z, z], dim=-1)
                g_t = torch.stack([z, g_img, z], dim=-1)
                b_t = torch.stack([z, z, b_img], dim=-1)
                top = torch.cat([r_t, g_t], dim=1)
                bot = torch.cat([b_t, rgb], dim=1)
                grid = torch.cat([top, bot], dim=0)
                grid_arr = (grid.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(grid_arr, mode="RGB").save(
                    os.path.join(save_dir, f"{view_idx:03d}_{tag}_ALL.png")
                )

                view_idx += 1
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"  [{view_idx}/{args.n_angles * len(args.elevations)}] "
                      f"el={el:+.0f}° az={az:.0f}°")

        print(f"\nDone — {view_idx} views saved to {save_dir}/")
        print(f"  Per view: R.png, G.png, B.png (greyscale channels)")
        print(f"            RGB.png (MIP-composited colour)")
        print(f"            ALL.png (2×2 tinted grid)")
        return  # exit without starting interactive viewer

    # Viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    actual_port = server._websock_server.port if hasattr(server._websock_server, 'port') else args.port

    # HTTPS proxy (start AFTER viser so we know the actual port)
    if not args.no_https:
        setup_https_proxy(actual_port, args.https_port)

    c = centroid.cpu().numpy()
    server.initial_camera.position = (c[0], c[1], c[2] + cam_dist)
    server.initial_camera.look_at = (float(c[0]), float(c[1]), float(c[2]))
    server.initial_camera.fov = fov_y
    server.initial_camera.up_direction = (0.0, 0.0, 1.0)

    status = server.gui.add_markdown(
        f"**PLY Viewer** | {N:,} splats | {IMG_W}x{IMG_H}"
    )

    # Render mode selector GUI
    with server.gui.add_folder("Rendering"):
        channel_select = server.gui.add_dropdown(
            "Mode",
            options=["Compare", "Alpha-Blend", "MIP RGB", "MIP R", "MIP G", "MIP B", "MIP Lum", "MIP All"],
            initial_value="Compare",
        )

    # Warm-up
    R0, T0 = cam_to_RT(server.initial_camera, device)
    vm0 = cam_to_viewmat(server.initial_camera, device)
    init_frame = render_view(R0, T0, vm0, channel_select.value)
    if device.type == "cuda":
        torch.cuda.synchronize()
    server.scene.set_background_image(init_frame, format="jpeg", jpeg_quality=90)
    print("Warm-up render done.")

    # Shared state for last camera handle per client
    client_cams = {}

    def do_render(client):
        cam_handle = client_cams.get(client.client_id)
        if cam_handle is None:
            return
        t0 = time.time()
        R, T = cam_to_RT(cam_handle, device)
        vm = cam_to_viewmat(cam_handle, device)
        mode = channel_select.value
        frame = render_view(R, T, vm, mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        client.scene.set_background_image(frame, format="jpeg", jpeg_quality=85)
        status.content = (
            f"**PLY Viewer** | {N:,} splats | {IMG_W}x{IMG_H} | "
            f"{mode} | {ms:.0f} ms ({1000 / max(ms, 1):.0f} FPS)"
        )

    # Re-render all clients when channel selection changes
    @channel_select.on_update
    def _(_) -> None:
        for client in server.get_clients().values():
            do_render(client)

    @server.on_client_connect
    def on_connect(client: viser.ClientHandle) -> None:
        @client.camera.on_update
        def _(cam_handle) -> None:
            client_cams[client.client_id] = cam_handle
            do_render(client)

    print(f"\n{'=' * 50}")
    print(f"  Interactive Gaussian Splat Viewer")
    print(f"{'=' * 50}")
    if not args.no_https:
        print(f"  HTTPS: https://localhost:{args.https_port}")
    print(f"  HTTP:  http://localhost:{actual_port}")
    print(f"  {N:,} splats | {IMG_W}x{IMG_H} | Alpha-Blend + MIP (β={BETA})")
    print(f"  Ctrl+C to stop")
    print(f"{'=' * 50}\n")

    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()


if __name__ == "__main__":
    main()
