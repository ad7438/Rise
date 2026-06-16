#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements_server.txt"

echo "[1/5] Basic environment checks"
python -V
nvidia-smi || true
df -h

echo "[2/5] Install basic system tools"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git tmux unzip

echo "[3/5] Upgrade pip"
python -m pip install --upgrade pip setuptools wheel

echo "[4/5] Install Python dependencies"
python -m pip install -r "$REQ_FILE"

echo "[5/5] Verify key imports"
python - <<'PY'
mods = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "PIL",
    "cv2",
    "matplotlib",
    "pandas",
    "tqdm",
    "transformers",
    "skimage",
    "sklearn",
    "py_sod_metrics",
    "tensorboardX",
    "thop",
    "omegaconf",
]
failed = []
for name in mods:
    try:
        __import__(name)
        print(f"OK: {name}")
    except Exception as exc:
        failed.append((name, repr(exc)))
        print(f"FAIL: {name} -> {exc}")

if failed:
    raise SystemExit(f"Import check failed for: {failed}")
PY

echo
echo "Setup finished."
echo "Next suggested commands:"
echo "  cd /workspace/RISE-master"
echo "  tmux new -s rise"
