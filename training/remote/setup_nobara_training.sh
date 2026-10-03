#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Ortho4XP-SegFormer}"
DATA_ROOT="${DATA_ROOT:-$HOME/model_training}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-nobara-training}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "[nobara-setup] project: $PROJECT_DIR"
echo "[nobara-setup] data:    $DATA_ROOT"
echo "[nobara-setup] venv:    $VENV_DIR"

mkdir -p "$PROJECT_DIR" "$DATA_ROOT"/{manifests,runs,labels,tiles}

if ! command -v python3 >/dev/null 2>&1; then
  echo "[nobara-setup] ERROR: python3 is not installed or not on PATH" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[nobara-setup] ERROR: nvidia-smi not found; install/verify Nobara NVIDIA drivers first" >&2
  exit 1
fi

nvidia-smi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install --only-binary=:all: \
  numpy \
  pillow \
  opencv-python \
  pyproj \
  requests \
  shapely \
  "ultralytics>=8.0.0"

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"cuda_version={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}")
PY

echo "[nobara-setup] ready"
