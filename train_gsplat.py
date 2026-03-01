#!/usr/bin/env python3
"""
train_gsplat.py — Train 3D Gaussian Splatting on GT MIP projections using gsplat
=================================================================================

Uses the gsplat library (rasterization with alpha-compositing) to fit 3D Gaussians
to the ground-truth MIP projection images from gt_mip_dataset/.

Usage:
    python train_gsplat.py                          # defaults
    python train_gsplat.py --n_gaussians 50000 --n_steps 10000
    python train_gsplat.py --resume checkpoints/gsplat_ckpt/gsplat_step5000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import gsplat


# =====================================================================
#  Dataset: load GT MIP images + camera parameters
# =====================================================================
class MIPDataset:
    """Loads GT MIP projections and camera parameters from gt_mip_dataset/."""

    def __init__(self, data_dir: str, device: torch.device):
        data_dir = Path(data_dir)
        with open(data_dir / "cameras.json") as f:
            meta = json.load(f)

        intr = meta["intrinsics"]
        self.width = intr["width"]
        self.height = intr["height"]
        self.near = intr["near"]
        self.far = intr["far"]

        # Intrinsics matrix (3, 3)
        self.K = torch.tensor([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=device)

        # Load all frames
        self.images = []    # list of (H, W) float32 tensors in [0, 1]
        self.viewmats = []  # list of (4, 4) float32 tensors

        for frame in meta["frames"]:
            # Load 16-bit grayscale → normalise to [0, 1]
            img_path = data_dir / frame["file"]
            img_np = np.array(Image.open(img_path)).astype(np.float32)
            img_np /= img_np.max() + 1e-8  # normalise per-image
            img_t = torch.from_numpy(img_np).to(device)
            self.images.append(img_t)

            # Build 4×4 world-to-camera matrix
            R = torch.tensor(frame["R"], dtype=torch.float32, device=device)
            T = torch.tensor(frame["T"], dtype=torch.float32, device=device)
            viewmat = torch.eye(4, dtype=torch.float32, device=device)
            viewmat[:3, :3] = R
            viewmat[:3, 3] = T
            self.viewmats.append(viewmat)

        self.n_views = len(self.images)
        print(f"  Loaded {self.n_views} views, {self.width}×{self.height}")

    def __len__(self):
        return self.n_views

    def get_batch(self, indices: list[int]):
        """Return stacked viewmats (C,4,4), Ks (C,3,3), images (C,H,W)."""
        viewmats = torch.stack([self.viewmats[i] for i in indices])
        Ks = self.K.unsqueeze(0).expand(len(indices), -1, -1)
        imgs = torch.stack([self.images[i] for i in indices])
        return viewmats, Ks, imgs


# =====================================================================
#  Gaussian model (parameter container)
# =====================================================================
class GaussianModel:
    """Manages trainable Gaussian parameters for gsplat."""

    def __init__(self, n_gaussians: int, device: torch.device, scene_extent: float = 1.0):
        self.device = device
        self.n = n_gaussians

        # Initialise means uniformly in [-extent, extent]^3
        self.means = (torch.rand(n_gaussians, 3, device=device) * 2 - 1) * scene_extent
        self.means.requires_grad_(True)

        # Quaternions (wxyz) — identity + small noise
        self.quats = torch.zeros(n_gaussians, 4, device=device)
        self.quats[:, 0] = 1.0
        self.quats += torch.randn_like(self.quats) * 0.01
        self.quats = F.normalize(self.quats, dim=-1)
        self.quats.requires_grad_(True)

        # Log-scales → exp gives actual scales
        init_log_scale = math.log(scene_extent / (n_gaussians ** (1/3)))
        self.log_scales = torch.full((n_gaussians, 3), init_log_scale, device=device)
        self.log_scales.requires_grad_(True)

        # Logit-opacities → sigmoid gives [0, 1] opacities
        self.logit_opacities = torch.full((n_gaussians,), 0.5, device=device)
        self.logit_opacities.requires_grad_(True)

        # Colors (grayscale intensity as RGB) — init to mid-grey
        self.log_colors = torch.zeros(n_gaussians, 3, device=device)
        self.log_colors.requires_grad_(True)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device):
        """Load from a saved gsplat checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device)
        n = ckpt["means"].shape[0]
        model = cls.__new__(cls)
        model.device = device
        model.n = n
        model.means = ckpt["means"].to(device).requires_grad_(True)
        model.quats = ckpt["quats"].to(device).requires_grad_(True)
        model.log_scales = ckpt["log_scales"].to(device).requires_grad_(True)
        model.logit_opacities = ckpt["logit_opacities"].to(device).requires_grad_(True)
        model.log_colors = ckpt["log_colors"].to(device).requires_grad_(True)
        return model, ckpt.get("step", 0)

    def param_groups(self, lr_base: float = 3e-3) -> list[dict]:
        """Return param groups with per-parameter learning rates."""
        return [
            {"params": [self.means],            "lr": lr_base,      "name": "means"},
            {"params": [self.quats],             "lr": lr_base,      "name": "quats"},
            {"params": [self.log_scales],        "lr": lr_base * 0.5, "name": "log_scales"},
            {"params": [self.logit_opacities],   "lr": lr_base * 2,  "name": "logit_opacities"},
            {"params": [self.log_colors],        "lr": lr_base,      "name": "log_colors"},
        ]

    def get_params(self):
        """Return activated parameters for gsplat.rasterization."""
        scales = torch.exp(self.log_scales).clamp(1e-6, 1.0)
        opacities = torch.sigmoid(self.logit_opacities)
        colors = torch.sigmoid(self.log_colors)          # (N, 3) RGB
        quats = F.normalize(self.quats, dim=-1)
        return self.means, quats, scales, opacities, colors

    def state_dict(self, step: int = 0):
        return {
            "means": self.means.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "log_scales": self.log_scales.detach().cpu(),
            "logit_opacities": self.logit_opacities.detach().cpu(),
            "log_colors": self.log_colors.detach().cpu(),
            "step": step,
        }


# =====================================================================
#  Rendering wrapper
# =====================================================================
def render_batch(
    model: GaussianModel,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 10.0,
    bg_color: float = 0.0,
) -> Tensor:
    """
    Render C views at once using gsplat.rasterization.
    Returns (C, H, W, 3) rendered images in [0, 1].
    """
    means, quats, scales, opacities, colors = model.get_params()

    renders, alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        render_mode="RGB",
        packed=True,
        rasterize_mode="antialiased",
    )
    return renders, alphas, meta


# =====================================================================
#  Loss functions
# =====================================================================
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()


def ssim_loss(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Simple SSIM on (B, C, H, W) tensors."""
    C = pred.shape[1]
    kernel_1d = torch.ones(window_size, device=pred.device) / window_size
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, -1, -1)

    pad = window_size // 2
    mu_pred = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_tgt  = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_pred_sq = mu_pred ** 2
    mu_tgt_sq  = mu_tgt ** 2
    mu_cross   = mu_pred * mu_tgt

    sigma_pred = F.conv2d(pred ** 2, kernel, padding=pad, groups=C) - mu_pred_sq
    sigma_tgt  = F.conv2d(target ** 2, kernel, padding=pad, groups=C) - mu_tgt_sq
    sigma_cross = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_cross

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_tgt_sq + C1) * (sigma_pred + sigma_tgt + C2))

    return 1.0 - ssim_map.mean()


def psnr(pred: Tensor, target: Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


# =====================================================================
#  Training loop
# =====================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ──
    dataset = MIPDataset(args.data_dir, device)

    # ── Init or resume model ──
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        model, start_step = GaussianModel.from_checkpoint(args.resume, device)
        print(f"  Loaded {model.n} Gaussians at step {start_step}")
    else:
        model = GaussianModel(args.n_gaussians, device, scene_extent=1.0)
        print(f"  Initialised {model.n} Gaussians")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.param_groups(lr_base=args.lr))

    # ── LR scheduler (cosine annealing) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.lr * 0.01
    )

    # ── Output dir ──
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──
    all_indices = list(range(dataset.n_views))
    views_per_step = min(args.views_per_step, dataset.n_views)

    print(f"\n  Training: {args.n_steps} steps, {views_per_step} views/step")
    print(f"  Output: {out_dir}\n")

    t0 = time.time()
    for step in range(start_step, args.n_steps):
        # Sample random views
        batch_idx = np.random.choice(all_indices, size=views_per_step, replace=False).tolist()
        viewmats, Ks, gt_imgs = dataset.get_batch(batch_idx)

        # gt_imgs: (C, H, W) grayscale → (C, H, W, 3) RGB for loss
        gt_rgb = gt_imgs.unsqueeze(-1).expand(-1, -1, -1, 3)

        # Render
        renders, alphas, meta = render_batch(
            model, viewmats, Ks,
            dataset.width, dataset.height,
            near=dataset.near, far=dataset.far,
        )
        # renders: (C, H, W, 3)

        # Loss: L1 + 0.2 * SSIM
        # Convert to (C, 3, H, W) for SSIM
        pred_bchw = renders.permute(0, 3, 1, 2)
        gt_bchw = gt_rgb.permute(0, 3, 1, 2)

        loss_l1 = l1_loss(renders, gt_rgb)
        loss_ss = ssim_loss(pred_bchw, gt_bchw)
        loss = (1.0 - args.lambda_ssim) * loss_l1 + args.lambda_ssim * loss_ss

        # Scale regularisation
        scales = torch.exp(model.log_scales)
        scale_reg = (scales.clamp(max=0.5) - scales).abs().mean() * 0.001
        loss = loss + scale_reg

        # Backward + step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Normalize quaternions
        with torch.no_grad():
            model.quats.data = F.normalize(model.quats.data, dim=-1)

        # Logging
        if (step + 1) % args.log_every == 0 or step == 0:
            with torch.no_grad():
                psnr_val = psnr(renders, gt_rgb)
            elapsed = time.time() - t0
            steps_per_sec = (step - start_step + 1) / elapsed
            print(
                f"  Step {step+1:6d}/{args.n_steps} | "
                f"Loss {loss.item():.5f} (L1={loss_l1.item():.5f}, SSIM={loss_ss.item():.4f}) | "
                f"PSNR {psnr_val:.2f} dB | "
                f"K={model.n} | {steps_per_sec:.1f} it/s"
            )

        # Save checkpoint
        if (step + 1) % args.save_every == 0 or step + 1 == args.n_steps:
            ckpt_path = out_dir / f"gsplat_step{step+1}.pt"
            torch.save(model.state_dict(step=step + 1), str(ckpt_path))
            print(f"  → Saved {ckpt_path}")

    # ── Export final .ply ──
    print("\n  Exporting final .ply ...")
    means, quats, scales, opacities, colors = model.get_params()
    # gsplat export needs SH coefficients: sh0 = (N, 1, 3), shN = (N, K, 3)
    sh0 = colors.unsqueeze(1).detach()  # (N, 1, 3) — DC band only
    # shN must be (N, num_higher_sh, 3) — use zeros for no higher-order SH
    shN = torch.zeros(means.shape[0], 0, 3, device=means.device)
    ply_path = out_dir / "gsplat_final.ply"
    gsplat.export_splats(
        means=means.detach(),
        scales=scales.detach(),
        quats=quats.detach(),
        opacities=opacities.detach(),
        sh0=sh0,
        shN=shN,
        save_to=str(ply_path),
    )
    print(f"  → Exported {ply_path}  ({ply_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\nDone!")


# =====================================================================
#  CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train 3DGS on GT MIP projections using gsplat")
    parser.add_argument("--data_dir", default="gt_mip_dataset", help="Path to gt_mip_dataset/")
    parser.add_argument("--out_dir", default="checkpoints/gsplat_ckpt", help="Checkpoint output dir")
    parser.add_argument("--n_gaussians", type=int, default=50000, help="Initial number of Gaussians")
    parser.add_argument("--n_steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--views_per_step", type=int, default=4, help="Views per optimizer step")
    parser.add_argument("--lr", type=float, default=3e-3, help="Base learning rate")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--log_every", type=int, default=50, help="Log interval")
    parser.add_argument("--save_every", type=int, default=2000, help="Save interval")
    parser.add_argument("--densify_until", type=int, default=5000, help="Densify until this step")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
