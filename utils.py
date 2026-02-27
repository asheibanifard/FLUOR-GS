# ===================================================================
#  Utilities
# ===================================================================
from datetime import datetime
import logging
import os

import numpy as np
import torch
import yaml
import tifffile as tiff

def load_config(path: str = "config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{ts}.log")

    # Reset root logger to avoid duplicate handlers across runs
    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger("gmf_train")
    logger.info(f"Log → {log_file}")
    return logger


def load_tif_data(file_path: str) -> np.ndarray:
    """Load a 3-D TIFF and normalise to [0, 1] float32."""
    vol = tiff.imread(file_path).astype(np.float32)  # (Z, Y, X)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def load_swc(file_path: str) -> np.ndarray:
    """
    Load an SWC morphology file and return (N, 4) array: [x, y, z, radius].
    SWC format: id  type  x  y  z  radius  parent_id
    Skips comment lines starting with '#'.
    """
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                x, y, z, r = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                rows.append([x, y, z, r])
    if not rows:
        raise ValueError(f"No valid SWC nodes found in {file_path}")
    return np.array(rows, dtype=np.float32)


def swc_to_normalised_coords(
    swc_data: np.ndarray,
    vol_shape: tuple[int, int, int],
    bounds: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert SWC coordinates (in voxel space) to normalised [-1, 1] coords.

    Args:
        swc_data: (N, 4) array [x, y, z, radius] in voxel coordinates.
        vol_shape: (Z, Y, X) shape of the volume.
        bounds: [[xlo,xhi],[ylo,yhi],[zlo,zhi]] normalised bounds (default [-1,1]).

    Returns:
        coords: (N, 3) normalised [x, y, z] coords.
        radii:  (N,) normalised radii (average of xyz scale factors).
    """
    if bounds is None:
        bounds = [[-1, 1], [-1, 1], [-1, 1]]
    Z, Y, X = vol_shape
    # SWC x → volume X axis, y → Y, z → Z
    vox_max = np.array([X - 1, Y - 1, Z - 1], dtype=np.float32)
    vox_max = np.maximum(vox_max, 1.0)  # avoid /0

    xyz = swc_data[:, :3]  # (N, 3) in voxel coords
    # Normalise each axis to [0, 1] then map to bounds
    norm01 = xyz / vox_max  # (N, 3) in [0, 1]
    coords = np.zeros_like(norm01)
    scale_factors = []
    for i in range(3):
        lo, hi = bounds[i][0], bounds[i][1]
        coords[:, i] = norm01[:, i] * (hi - lo) + lo
        scale_factors.append((hi - lo) / vox_max[i])

    # Normalise radius: average scale factor across axes
    avg_scale = np.mean(scale_factors)
    radii = swc_data[:, 3] * avg_scale

    return coords.astype(np.float32), radii.astype(np.float32)

# ===================================================================
#  Weight schedule
# ===================================================================
def weight_schedule(cfg: dict, step: int, total: int) -> tuple[float, float]:
    sch = cfg["training"].get("weight_schedule", "constant").lower()
    if sch == "constant":
        return (
            float(cfg["training"].get("w_vol", 1.0)),
            float(cfg["training"].get("w_mip", 1.0)),
        )

    vs = float(cfg["training"].get("w_vol_start", 1.0))
    ms = float(cfg["training"].get("w_mip_start", 0.1))
    ve = float(cfg["training"].get("w_vol_end", 1.0))
    me = float(cfg["training"].get("w_mip_end", 1.0))
    tf = float(cfg["training"].get("weight_transition_fraction", 0.3))

    t = step / max(1, total - 1)

    if sch == "step":
        return (vs, ms) if t < tf else (ve, me)

    if sch == "linear_ramp":
        if t < tf:
            return vs, ms
        r = (t - tf) / max(1e-12, 1.0 - tf)
        return vs + (ve - vs) * r, ms + (me - ms) * r

    return float(cfg["training"].get("w_vol", 1.0)), float(
        cfg["training"].get("w_mip", 1.0)
    )

# ===================================================================
#  MIP helpers
# ===================================================================
def mip_teacher_z(vol: np.ndarray) -> np.ndarray:
    """Ground-truth z-axis Maximum Intensity Projection."""
    return vol.max(axis=0).astype(np.float32)


def sample_pixels_from_mip(mip: np.ndarray, num_samples: int):
    Y, X = mip.shape
    Npix = Y * X
    if num_samples > Npix:
        raise ValueError(f"num_samples={num_samples} > total pixels={Npix}")
    idx = np.random.choice(Npix, size=num_samples, replace=False)
    y, x = idx // X, idx % X
    xn = (x / max(X - 1, 1)) * 2 - 1
    yn = (y / max(Y - 1, 1)) * 2 - 1
    xy = torch.from_numpy(np.stack([xn, yn], axis=1)).float()
    t = torch.from_numpy(mip[y, x]).float()
    return xy, t


def compute_tau_schedule(tau_start: float, tau_end: float, t: float) -> float:
    """Anneal soft-max temperature.  t ∈ [0, 1]."""
    return float(tau_start * (tau_end / max(tau_start, 1e-12)) ** t)
