#!/bin/bash
set -e

echo "============================================"
echo "  vast.ai Instance Setup Script"
echo "  Conda + CUDA + Jupyter Kernel Setup"
echo "============================================"

# ---- System-level dependencies ----
echo "[1/7] Installing system dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tmux \
    htop \
    nvtop \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ---- CUDA build tools (headers, nvcc, etc.) ----
echo "[2/7] Ensuring CUDA dev toolkit is available..."
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found — installing CUDA toolkit..."
    # Adjust version as needed; 12.1 is a safe default for most vast.ai images
    CUDA_VER=${CUDA_VERSION:-12.1}
    conda install -y -c nvidia cuda-toolkit=${CUDA_VER} 2>/dev/null || \
    pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 --quiet
else
    echo "nvcc found: $(nvcc --version | grep release)"
fi

# ---- Conda environment ----
ENV_NAME="neurogs"
PYTHON_VER="3.10"

echo "[3/7] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})..."
if command -v conda &> /dev/null; then
    echo "Conda found at: $(which conda)"
else
    echo "Conda not found — installing Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p $HOME/miniforge3
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniforge.sh
fi

# Source conda for current shell
eval "$(conda shell.bash hook)"

conda create -y -n ${ENV_NAME} python=${PYTHON_VER}
conda activate ${ENV_NAME}

# ---- Core Python packages ----
echo "[4/7] Installing core Python packages..."
pip install --upgrade pip setuptools wheel

# PyTorch — match CUDA version on the instance
CUDA_TAG=$(python -c "
import subprocess, re
out = subprocess.check_output(['nvcc','--version']).decode()
m = re.search(r'release (\d+)\.(\d+)', out)
if m: print(f'cu{m.group(1)}{m.group(2)}')
else: print('cu121')
" 2>/dev/null || echo "cu121")

echo "   Detected CUDA tag: ${CUDA_TAG}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_TAG}

# ---- CUDA kernel build dependencies ----
echo "[5/7] Installing CUDA kernel / extension build deps..."
pip install \
    ninja \
    pybind11 \
    jaxtyping \
    beartype

# 3D Gaussian Splatting ecosystem
pip install \
    plyfile \
    diff-gaussian-rasterization 2>/dev/null || echo "  [info] diff-gaussian-rasterization not pip-installable; will need manual build"
pip install \
    simple-knn 2>/dev/null || echo "  [info] simple-knn not pip-installable; will need manual build"
pip install \
    nerfacc \
    gsplat 2>/dev/null || echo "  [info] gsplat install skipped — install manually if needed"

# ---- Research / ML stack ----
echo "[6/7] Installing ML & research packages..."
pip install \
    numpy \
    scipy \
    scikit-image \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    tqdm \
    wandb \
    tensorboard \
    einops \
    timm \
    omegaconf \
    hydra-core \
    lpips \
    pytorch-msssim \
    open3d \
    trimesh \
    nibabel \
    SimpleITK \
    tifffile \
    h5py \
    zarr \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    pillow

# ---- Jupyter / IPython kernel ----
echo "[7/7] Setting up Jupyter kernel..."
pip install ipykernel jupyterlab notebook ipywidgets

python -m ipykernel install \
    --user \
    --name ${ENV_NAME} \
    --display-name "Python (${ENV_NAME})"

# Also install in system jupyter if present
python -m ipykernel install \
    --prefix=/usr/local \
    --name ${ENV_NAME} \
    --display-name "Python (${ENV_NAME})" 2>/dev/null || true

# ---- Verify installation ----
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'cuDNN:    {torch.backends.cudnn.version()}')
print(f'GPUs:     {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  [{i}] {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
print(f'CUDA kernel compilation: ', end='')
# Quick sanity check that torch extensions can compile
try:
    from torch.utils.cpp_extension import load_inline
    print('OK (inline ext available)')
except Exception as e:
    print(f'WARNING — {e}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "  Jupyter kernel:  'Python (${ENV_NAME})'"
echo "============================================"
