#!/usr/bin/env python3
"""
end2end.py — End-to-End MIP Splatting Training
===============================================

Trains 3D Gaussians from scratch (no pretrained model) by:
  1. Initialising K Gaussians randomly or from SWC skeleton
  2. Rendering GT MIP projections from the raw volume at M viewpoints
  3. Splatting the Gaussians to 2D → differentiable soft-MIP images
  4. Computing loss (weighted MSE + SSIM + edge + scale reg)
  5. Back-propagating through the entire splatting pipeline
  6. Updating (means, log_scales, quaternions, log_intensities)
  7. Repeating until convergence

All rendering mechanics are imported from rendering.py — no duplication.

Usage:
    cd /workspace/hisnegs/src/renderer
    python end2end.py                           # uses config_splat.yml
    python end2end.py --config my_config.yml    # custom config
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── Import all rendering primitives from rendering.py ────────────────
from rendering import (
    Camera,
    GaussianParameters,
    compute_aspect_scales,
    apply_aspect_correction,
    render_mip_projection,
    render_gt_mip,
    generate_camera_poses,
    generate_mip_dataset,
    load_config,
    load_volume,
    _orbit_pose,
    weighted_mse_loss,
    ssim_loss_fn,
    edge_loss,
    psnr_metric,
    transform_to_camera,
    project_to_2d,
    splat_mip_grid,
)

# ── SWC utilities from parent package ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_swc, swc_to_normalised_coords


# =====================================================================
#  Gaussian parameter initialisation
# =====================================================================
def init_gaussians(
    K:          int,
    init_scale: float     = 0.09,
    init_amp:   float     = 0.05,
    bounds:     list      = None,
    swc_path:   str | None = None,
    vol_shape:  tuple     = None,
    device:     str       = "cuda",
) -> Tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
    """
    Create fresh Gaussian parameters.

    Returns
    -------
    means           : (K, 3)  nn.Parameter
    log_scales      : (K, 3)  nn.Parameter   (isotropic at init)
    quaternions     : (K, 4)  nn.Parameter   (identity rotation)
    log_intensities : (K,)    nn.Parameter
    """
    if bounds is None:
        bounds = [[-1, 1], [-1, 1], [-1, 1]]

    # ── Means: from SWC morphology or random ─────────────────────
    swc_coords = None
    if swc_path and os.path.exists(swc_path):
        swc_data = load_swc(swc_path)
        swc_coords, swc_radii = swc_to_normalised_coords(
            swc_data, vol_shape, bounds=bounds)
        print(f"  SWC init: {swc_data.shape[0]} nodes from {swc_path}")

    if swc_coords is not None:
        n_swc = swc_coords.shape[0]
        if n_swc >= K:
            idx = np.linspace(0, n_swc - 1, K, dtype=int)
            means = torch.from_numpy(swc_coords[idx]).float().to(device)
        else:
            means_swc = torch.from_numpy(swc_coords).float()
            n_extra = K - n_swc
            pair_idx = torch.randint(0, max(n_swc - 1, 1), (n_extra,))
            t_interp = torch.rand(n_extra, 1)
            lo_idx = pair_idx
            hi_idx = (pair_idx + 1).clamp(max=n_swc - 1)
            extra = means_swc[lo_idx] * (1 - t_interp) + means_swc[hi_idx] * t_interp
            extra += torch.randn_like(extra) * 0.001
            means = torch.cat([means_swc, extra], dim=0).to(device)
        print(f"  SWC → {K} Gaussian means")
    else:
        means = torch.zeros(K, 3, device=device)
        for i in range(3):
            lo, hi = bounds[i][0], bounds[i][1]
            means[:, i] = torch.rand(K, device=device) * (hi - lo) + lo
        print(f"  Random init within {bounds}")

    # ── Scales, rotations, intensities ───────────────────────────
    log_scales = torch.ones(K, 3, device=device) * math.log(init_scale)
    quaternions = torch.zeros(K, 4, device=device)
    quaternions[:, 0] = 1.0  # identity rotation
    log_intensities = torch.ones(K, device=device) * math.log(max(init_amp, 1e-6))

    print(f"  K={K}  init_scale={init_scale:.4f}  init_amp={init_amp:.4f}")

    return (
        nn.Parameter(means),
        nn.Parameter(log_scales),
        nn.Parameter(quaternions),
        nn.Parameter(log_intensities),
    )


# =====================================================================
#  Build covariance + GaussianParameters from raw params
# =====================================================================
def build_gaussians(
    means:           torch.Tensor,
    log_scales:      torch.Tensor,
    quaternions:     torch.Tensor,
    log_intensities: torch.Tensor,
    aspect_scales:   Optional[torch.Tensor] = None,
) -> GaussianParameters:
    """
    Differentiable construction: raw params → GaussianParameters.

    This function stays in the autograd graph so gradients flow back
    through covariances → log_scales / quaternions.
    """
    K = means.shape[0]
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)  # (K, 3)

    # Quaternion → rotation matrix
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(K, 3, 3, device=means.device, dtype=means.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - w*z);  R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);      R[:, 1, 1] = 1 - 2*(x*x+z*z);R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);      R[:, 2, 1] = 2*(y*z + w*x);  R[:, 2, 2] = 1 - 2*(x*x+y*y)

    S2  = torch.diag_embed(scales ** 2)           # (K, 3, 3)
    cov = R @ S2 @ R.transpose(-2, -1)            # Σ = R diag(s²) Rᵀ

    intensities = torch.sigmoid(log_intensities)   # ∈ (0, 1)

    g = GaussianParameters(means=means, covariances=cov, intensities=intensities)
    if aspect_scales is not None:
        g = apply_aspect_correction(g, aspect_scales)
    return g


# =====================================================================
#  Single-view forward  (differentiable)
# =====================================================================
def render_view(
    gaussians: GaussianParameters,
    camera:    Camera,
    R:         torch.Tensor,
    T:         torch.Tensor,
    beta:      float = 50.0,
    chunk_size: int  = 4096,
) -> Tuple[torch.Tensor, int]:
    """Thin wrapper around render_mip_projection for clarity."""
    return render_mip_projection(
        gaussians, camera, R, T, beta=beta, chunk_size=chunk_size)


# =====================================================================
#  Per-view loss
# =====================================================================
def compute_view_loss(
    pred:            torch.Tensor,
    target:          torch.Tensor,
    log_scales:      torch.Tensor,
    fg_weight:       float = 5.0,
    lambda_ssim:     float = 0.2,
    lambda_edge:     float = 0.1,
    lambda_scale:    float = 0.001,
    scale_min:       float = 0.005,
    lambda_scale_max: float = 0.01,
    scale_max:       float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """
    Composite loss for one rendered view vs GT.

    Returns
    -------
    loss    : scalar tensor  (grad-enabled)
    metrics : dict of floats for logging
    """
    # Weighted MSE (foreground emphasis)
    mse = weighted_mse_loss(pred, target, fg_weight=fg_weight)

    # SSIM (structural similarity)
    ssim_l = ssim_loss_fn(pred, target)
    ssim_val = 1.0 - ssim_l.item()

    # Edge / gradient loss
    edge_l = edge_loss(pred, target) if lambda_edge > 0 else torch.tensor(0.0, device=pred.device)

    # Scale regularisation
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    sr_small = lambda_scale * torch.clamp(scale_min - scales, min=0.0).mean()
    sr_big   = lambda_scale_max * torch.clamp(scales - scale_max, min=0.0).mean()
    scale_reg = sr_small + sr_big

    loss = mse + lambda_ssim * ssim_l + lambda_edge * edge_l + scale_reg

    # Unweighted PSNR for reporting
    mse_unw = F.mse_loss(pred, target)
    psnr = -10.0 * torch.log10(mse_unw.clamp(min=1e-12))

    return loss, {
        "loss": loss.item(),
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim_val,
        "scale_reg": scale_reg.item(),
    }


# =====================================================================
#  Densification & Pruning
# =====================================================================
def densify_and_prune(
    means, log_scales, quaternions, log_intensities,
    grad_accum, grad_count,
    grad_thresh:    float = 2e-6,
    scale_thresh:   float = 0.01,
    prune_thresh:   float = 0.01,
    min_gaussians:  int   = 2000,
    max_gaussians:  int   = 500000,
    split_factor:   float = 1.6,
):
    """
    Adaptive density control — split, clone, and prune.

    Modifies parameters in-place and returns new nn.Parameters.
    """
    device = means.device
    K = means.shape[0]
    avg_grad = grad_accum / grad_count.clamp(min=1)

    scales = torch.exp(log_scales.data).clamp(1e-5, 1e2)
    max_s  = scales.max(dim=-1).values
    intens = torch.sigmoid(log_intensities.data)

    high_grad  = avg_grad > grad_thresh
    split_mask = high_grad & (max_s > scale_thresh)
    clone_mask = high_grad & (~split_mask)

    new_m, new_ls, new_q, new_li = [], [], [], []

    # Clone small Gaussians
    if clone_mask.any():
        new_m.append(means.data[clone_mask])
        new_ls.append(log_scales.data[clone_mask])
        new_q.append(quaternions.data[clone_mask])
        new_li.append(log_intensities.data[clone_mask])

    # Split large Gaussians
    n_split = int(split_mask.sum().item())
    if n_split > 0:
        m  = means.data[split_mask]
        ls = log_scales.data[split_mask]
        q  = quaternions.data[split_mask]
        li = log_intensities.data[split_mask]
        s  = torch.exp(ls)

        # Compute principal axis from quaternion → rotation
        qn = F.normalize(q, p=2, dim=-1)
        ww, xx, yy, zz = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        Rm = torch.zeros(n_split, 3, 3, device=device)
        Rm[:, 0, 0] = 1-2*(yy*yy+zz*zz); Rm[:, 0, 1] = 2*(xx*yy-ww*zz); Rm[:, 0, 2] = 2*(xx*zz+ww*yy)
        Rm[:, 1, 0] = 2*(xx*yy+ww*zz);   Rm[:, 1, 1] = 1-2*(xx*xx+zz*zz); Rm[:, 1, 2] = 2*(yy*zz-ww*xx)
        Rm[:, 2, 0] = 2*(xx*zz-ww*yy);   Rm[:, 2, 1] = 2*(yy*zz+ww*xx);   Rm[:, 2, 2] = 1-2*(xx*xx+yy*yy)

        imax = s.argmax(dim=-1)
        bi = torch.arange(n_split, device=device)
        principal = Rm[bi, :, imax]
        offset = s[bi, imax].unsqueeze(-1) * 0.5

        child_ls = ls - math.log(split_factor)
        child_li = li - math.log(2.0)

        for sign in (1.0, -1.0):
            new_m.append(m + sign * principal * offset)
            new_ls.append(child_ls.clone())
            new_q.append(q.clone())
            new_li.append(child_li.clone())

    # Prune dim Gaussians
    keep = intens > prune_thresh
    if keep.sum() < min_gaussians:
        keep = torch.ones(K, dtype=torch.bool, device=device)
    n_pruned = int((~keep).sum().item())

    all_m  = torch.cat([means.data[keep]]           + new_m,  dim=0)
    all_ls = torch.cat([log_scales.data[keep]]      + new_ls, dim=0)
    all_q  = torch.cat([quaternions.data[keep]]     + new_q,  dim=0)
    all_li = torch.cat([log_intensities.data[keep]] + new_li, dim=0)

    # Enforce cap
    if all_m.shape[0] > max_gaussians:
        top_amps = torch.sigmoid(all_li)
        _, topk = top_amps.topk(max_gaussians, largest=True)
        topk = topk.sort()[0]
        all_m  = all_m[topk]
        all_ls = all_ls[topk]
        all_q  = all_q[topk]
        all_li = all_li[topk]

    n_clone = int(clone_mask.sum().item())
    K_new = all_m.shape[0]
    print(f"  [Densify] K: {K} → {K_new}  "
          f"(split {n_split}, clone {n_clone}, pruned {n_pruned})")

    return (
        nn.Parameter(all_m),
        nn.Parameter(all_ls),
        nn.Parameter(all_q),
        nn.Parameter(all_li),
    )


# =====================================================================
#  Main training loop
# =====================================================================
def train_end_to_end(
    means:           nn.Parameter,
    log_scales:      nn.Parameter,
    quaternions:     nn.Parameter,
    log_intensities: nn.Parameter,
    dataset:         List[dict],
    camera:          Camera,
    aspect_scales:   torch.Tensor,
    cfg:             dict,
    save_dir:        str = "checkpoints/end2end",
) -> List[dict]:
    """
    End-to-end training loop.

    Parameters
    ----------
    means, log_scales, quaternions, log_intensities
        Fresh or resumed Gaussian parameters (nn.Parameter on device).
    dataset
        List of {'image': (H,W) tensor, 'R': (3,3), 'T': (3,), ...}
        produced by generate_mip_dataset().
    camera : Camera
    aspect_scales : (3,) tensor
    cfg : dict from config_splat.yml
    save_dir : checkpoint output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    device = means.device

    # ── Config ───────────────────────────────────────────────────
    t   = cfg["training"]
    l   = cfg["loss"]
    sc  = cfg["scale_clamp"]
    li  = cfg["log_intensity_clamp"]
    pr  = cfg["pruning"]
    dn  = cfg["densification"]

    n_epochs        = t["n_epochs"]
    lr              = t["lr"]
    lr_final        = t.get("lr_final", lr * 0.01)
    beta_init       = t.get("beta_mip_init", t["beta_mip"])
    beta_final      = t["beta_mip"]
    beta_warmup     = t.get("beta_warmup_epochs", 0)
    views_per_step  = t.get("views_per_step", 4)
    log_every       = t["log_every"]
    save_every      = t["save_every"]
    chunk_size      = t["chunk_size"]

    fg_weight       = l.get("fg_weight", 5.0)
    lambda_ssim     = l.get("lambda_ssim", 0.2)
    lambda_edge     = l.get("lambda_edge", 0.1)
    lambda_scale    = l["lambda_scale"]
    scale_min       = l["scale_min"]
    lambda_scale_max= l["lambda_scale_max"]
    scale_max       = l["scale_max"]

    log_scale_min   = sc["log_min"]
    log_scale_max   = sc["log_max"]
    log_intens_min  = li["min"]
    log_intens_max  = li["max"]

    prune_every     = pr["prune_every"]
    prune_thresh    = pr["intens_thresh"]
    min_gaussians   = pr["min_gaussians"]

    dens_every      = dn["densify_every"]
    dens_start      = dn["start_epoch"]
    dens_stop       = dn["stop_epoch"]
    dens_grad       = dn["grad_thresh"]
    dens_scale      = dn["scale_thresh"]
    max_gaussians   = dn["max_gaussians"]
    split_factor    = dn["split_factor"]

    # ── Train / val split ────────────────────────────────────────
    M = len(dataset)
    val_ratio = t.get("val_ratio", 0.1)
    n_val = max(1, int(round(M * val_ratio)))
    n_val = min(n_val, M - 1)
    idx = list(range(M))
    rnd = random.Random(42)
    rnd.shuffle(idx)
    val_idx = set(idx[:n_val])
    train_data = [dataset[i] for i in range(M) if i not in val_idx]
    val_data   = [dataset[i] for i in range(M) if i in val_idx]
    print(f"  Train views: {len(train_data)}  Val views: {len(val_data)}")

    # ── Optimizer ────────────────────────────────────────────────
    lr_mults = [1.0, 0.5, 0.3, 1.0]  # means, scales, quats, intensities

    def make_optimizer():
        return torch.optim.Adam([
            {"params": [means],           "lr": lr * lr_mults[0]},
            {"params": [log_scales],      "lr": lr * lr_mults[1]},
            {"params": [quaternions],     "lr": lr * lr_mults[2]},
            {"params": [log_intensities], "lr": lr * lr_mults[3]},
        ])

    optimizer = make_optimizer()

    # Gradient accumulator for densification
    grad_accum = torch.zeros(means.shape[0], device=device)
    grad_count = torch.zeros(means.shape[0], device=device)

    history = []
    best = {"psnr": 0.0, "loss": float("inf")}

    print(f"\n{'='*60}")
    print(f"End-to-End MIP Splatting Training")
    print(f"  Epochs: {n_epochs}  LR: {lr} → {lr_final}")
    print(f"  Beta:   {beta_init} → {beta_final}  (warmup {beta_warmup} ep)")
    print(f"  K = {means.shape[0]} Gaussians")
    print(f"  M = {len(train_data)} train views,  {len(val_data)} val views")
    print(f"  Views/step: {views_per_step}")
    print(f"  Densify: every {dens_every} ep  (epochs {dens_start}–{dens_stop})")
    print(f"  Prune:   every {prune_every} ep")
    print(f"  Save:    every {save_every} ep → {save_dir}/")
    print(f"{'='*60}\n")

    pbar = tqdm(range(1, n_epochs + 1), desc="Training", unit="ep")

    for epoch in pbar:
        # ── Cosine LR annealing ──────────────────────────────────
        frac = (epoch - 1) / max(n_epochs - 1, 1)
        cosine_lr = lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * frac))
        for pg, mult in zip(optimizer.param_groups, lr_mults):
            pg["lr"] = cosine_lr * mult

        # ── Beta warmup ──────────────────────────────────────────
        if beta_warmup > 0 and epoch <= beta_warmup:
            beta = beta_init + (beta_final - beta_init) * (epoch / beta_warmup)
        else:
            beta = beta_final

        # ── Densification ────────────────────────────────────────
        if (dens_every > 0
                and epoch % dens_every == 0
                and dens_start <= epoch <= dens_stop
                and grad_count.sum() > 0):
            means, log_scales, quaternions, log_intensities = densify_and_prune(
                means, log_scales, quaternions, log_intensities,
                grad_accum, grad_count,
                grad_thresh=dens_grad,
                scale_thresh=dens_scale,
                prune_thresh=prune_thresh,
                min_gaussians=min_gaussians,
                max_gaussians=max_gaussians,
                split_factor=split_factor,
            )
            optimizer = make_optimizer()
            grad_accum = torch.zeros(means.shape[0], device=device)
            grad_count = torch.zeros(means.shape[0], device=device)
            torch.cuda.empty_cache()

        # ── Prune only (outside densify window) ──────────────────
        elif (prune_every > 0 and epoch % prune_every == 0 and epoch > 0
              and not (dens_start <= epoch <= dens_stop)):
            intens = torch.sigmoid(log_intensities.data)
            keep = intens > prune_thresh
            n_before = means.shape[0]
            if keep.sum() >= min_gaussians and keep.sum() < n_before:
                means           = nn.Parameter(means.data[keep])
                log_scales      = nn.Parameter(log_scales.data[keep])
                quaternions     = nn.Parameter(quaternions.data[keep])
                log_intensities = nn.Parameter(log_intensities.data[keep])
                optimizer = make_optimizer()
                grad_accum = torch.zeros(means.shape[0], device=device)
                grad_count = torch.zeros(means.shape[0], device=device)
                tqdm.write(f"  [Prune @ {epoch}] {n_before} → {means.shape[0]}")

        # ── Mini-batch SGD over views ────────────────────────────
        view_indices = list(range(len(train_data)))
        random.shuffle(view_indices)

        epoch_metrics = []

        for batch_start in range(0, len(view_indices), views_per_step):
            batch_idx = view_indices[batch_start : batch_start + views_per_step]
            optimizer.zero_grad()

            for vi in batch_idx:
                view = train_data[vi]
                # Build Gaussians (differentiable)
                gaussians = build_gaussians(
                    means, log_scales, quaternions, log_intensities,
                    aspect_scales=aspect_scales)

                # Render this view
                pred, n_vis = render_view(
                    gaussians, camera,
                    view["R"].to(device), view["T"].to(device),
                    beta=beta, chunk_size=chunk_size)

                # Loss
                loss, metrics = compute_view_loss(
                    pred, view["image"],
                    log_scales,
                    fg_weight=fg_weight,
                    lambda_ssim=lambda_ssim,
                    lambda_edge=lambda_edge,
                    lambda_scale=lambda_scale,
                    scale_min=scale_min,
                    lambda_scale_max=lambda_scale_max,
                    scale_max=scale_max,
                )
                loss.backward()
                metrics["n_visible"] = n_vis
                epoch_metrics.append(metrics)

            # Average gradients over mini-batch
            B = len(batch_idx)
            if B > 1:
                for p in [means, log_scales, quaternions, log_intensities]:
                    if p.grad is not None:
                        p.grad.div_(B)

            torch.nn.utils.clip_grad_norm_(
                [means, log_scales, quaternions, log_intensities], max_norm=1.0)

            # Accumulate gradient magnitudes for densification
            if means.grad is not None:
                with torch.no_grad():
                    gn = means.grad.norm(dim=-1)
                    if gn.shape[0] == grad_accum.shape[0]:
                        grad_accum += gn
                        grad_count += 1
                    else:
                        grad_accum = gn.clone()
                        grad_count = torch.ones_like(gn)

            optimizer.step()

        # ── Post-step clamping ───────────────────────────────────
        with torch.no_grad():
            log_scales.data.clamp_(log_scale_min, log_scale_max)
            log_intensities.data.clamp_(log_intens_min, log_intens_max)
            means.data.clamp_(-1.0, 1.0)

        # ── Epoch summary ────────────────────────────────────────
        avg = {k: float(np.mean([m[k] for m in epoch_metrics]))
               for k in epoch_metrics[0] if k != "n_visible"}
        avg["n_visible"] = int(np.mean([m["n_visible"] for m in epoch_metrics]))
        history.append(avg)

        best["loss"] = min(best["loss"], avg["loss"])
        best["psnr"] = max(best["psnr"], avg["psnr"])

        pbar.set_postfix({
            "loss": f"{best['loss']:.5f}",
            "psnr": f"{best['psnr']:.2f}",
            "K": means.shape[0],
        })

        if epoch % log_every == 0:
            tqdm.write(
                f"  Ep {epoch:>4d}/{n_epochs}  "
                f"loss={avg['loss']:.5f}  psnr={avg['psnr']:.2f} dB  "
                f"ssim={avg['ssim']:.4f}  K={means.shape[0]}  "
                f"β={beta:.1f}  lr={cosine_lr:.2e}")

        # ── Validation ───────────────────────────────────────────
        if epoch % save_every == 0 and val_data:
            with torch.no_grad():
                gaussians = build_gaussians(
                    means, log_scales, quaternions, log_intensities,
                    aspect_scales=aspect_scales)
                val_psnrs = []
                for view in val_data:
                    pred, _ = render_view(
                        gaussians, camera,
                        view["R"].to(device), view["T"].to(device),
                        beta=beta, chunk_size=chunk_size)
                    val_psnrs.append(psnr_metric(pred, view["image"]))
                avg_val_psnr = float(np.mean(val_psnrs))
            tqdm.write(f"  Val PSNR @ {epoch}: {avg_val_psnr:.2f} dB")

        # ── Checkpoint ───────────────────────────────────────────
        if epoch % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"e2e_ep{epoch}.pt")
            torch.save({
                "means":           means.data.cpu(),
                "log_scales":      log_scales.data.cpu(),
                "quaternions":     quaternions.data.cpu(),
                "log_intensities": log_intensities.data.cpu(),
                "epoch":           epoch,
            }, ckpt_path)
            tqdm.write(f"  Checkpoint → {ckpt_path}")

        if means.device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    # Final save
    final_path = os.path.join(save_dir, "e2e_final.pt")
    torch.save({
        "means":           means.data.cpu(),
        "log_scales":      log_scales.data.cpu(),
        "quaternions":     quaternions.data.cpu(),
        "log_intensities": log_intensities.data.cpu(),
        "epoch":           n_epochs,
    }, final_path)
    print(f"\nFinal model → {final_path}  (K={means.shape[0]})")

    return history


# =====================================================================
#  CLI entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="End-to-End MIP Splatting")
    parser.add_argument("--config", default="config_splat.yml",
                        help="YAML config path (default: config_splat.yml)")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    cfg = load_config(cfg_path)
    print(f"Config: {cfg_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ── 1. Load volume ───────────────────────────────────────────
    vol_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), cfg["dataset"]["vol_path"]))
    print(f"\nLoading volume: {vol_path}")
    vol_np = load_volume(vol_path)
    Z, Y, X = vol_np.shape
    vol_gpu = torch.from_numpy(vol_np).to(device)
    print(f"  Shape (Z,Y,X): ({Z},{Y},{X})")

    aspect_scales = compute_aspect_scales((Z, Y, X))
    print(f"  Aspect scales: {aspect_scales.tolist()}")

    # ── 2. Initialise Gaussians ──────────────────────────────────
    init_cfg = cfg.get("init", {})
    K          = int(init_cfg.get("num_gaussians", 10000))
    init_scale = float(init_cfg.get("init_scale", 0.09))
    init_amp   = float(init_cfg.get("init_amplitude", 0.05))
    bounds     = init_cfg.get("bounds", [[-1, 1], [-1, 1], [-1, 1]])

    swc_path_raw = cfg["dataset"].get("swc_path")
    swc_path = None
    if swc_path_raw and swc_path_raw != "null":
        swc_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), swc_path_raw))
        if not os.path.exists(swc_path):
            print(f"  WARNING: SWC not found at {swc_path}, using random init")
            swc_path = None

    print(f"\nInitialising Gaussians (end-to-end, no pretrained model):")
    means, log_scales, quaternions, log_intensities = init_gaussians(
        K=K, init_scale=init_scale, init_amp=init_amp, bounds=bounds,
        swc_path=swc_path, vol_shape=(Z, Y, X), device=str(device))

    # ── 3. Camera ────────────────────────────────────────────────
    H, W = int(Y), int(X)
    # Use lower resolution for training speed (full res for final eval)
    train_res = int(cfg["training"].get("train_resolution", 256))
    camera = Camera.from_config(cfg, width=train_res, height=train_res)
    print(f"\nCamera: {train_res}×{train_res}  fx={camera.fx:.1f}")

    # ── 4. Generate GT MIP dataset ───────────────────────────────
    print("\nGenerating camera poses...")
    p = cfg["poses"]
    poses = generate_camera_poses(
        n_azimuth=p["n_azimuth"],
        n_elevation=p["n_elevation"],
        elevation_range=(p["elevation_min"], p["elevation_max"]),
        radius=p["radius"],
        include_axis_aligned=p["include_axis_aligned"],
    )
    print(f"  M = {len(poses)} projection views")

    rm = cfg["ray_marching"]
    # Render GT at training resolution
    gt_camera = Camera.from_config(cfg, width=train_res, height=train_res)
    print(f"\nRendering GT MIP dataset at {train_res}×{train_res}...")
    dataset = generate_mip_dataset(
        vol_gpu, gt_camera, poses,
        n_ray_samples=rm["n_samples"],
        near=rm["near"],
        far=rm["far"],
        aspect_scales=aspect_scales,
    )

    # ── 5. Train ─────────────────────────────────────────────────
    out_cfg = cfg["output"]
    save_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), out_cfg.get("mip_ckpt_dir", "checkpoints/end2end")))

    history = train_end_to_end(
        means=means,
        log_scales=log_scales,
        quaternions=quaternions,
        log_intensities=log_intensities,
        dataset=dataset,
        camera=camera,
        aspect_scales=aspect_scales,
        cfg=cfg,
        save_dir=save_dir,
    )

    # ── 6. Final validation renders ──────────────────────────────
    print("\nFinal validation renders...")
    gaussians = build_gaussians(
        means, log_scales, quaternions, log_intensities,
        aspect_scales=aspect_scales)

    for vi in [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]:
        view = dataset[vi]
        with torch.no_grad():
            pred, n_vis = render_view(
                gaussians, camera,
                view["R"].to(device), view["T"].to(device),
                beta=cfg["training"]["beta_mip"])
        p = psnr_metric(pred, view["image"])
        print(f"  View {vi:>3d}  PSNR={p:.2f} dB  visible={n_vis}")

    print("\n✓ End-to-end training complete!")


if __name__ == "__main__":
    main()
