# BitaHub Server Setup

This repository is intended to be synchronized as code only. Datasets,
checkpoints, predictions, logs, and generated visualizations should stay on the
server storage and must not be committed to Git.

## 1. Recommended BitaHub Configuration

- GPU: 1 x RTX 4090 24GB is enough for the current single-GPU scripts.
- Image: PyTorch + CUDA + cuDNN + Ubuntu 22.04, preferably Python 3.10 or 3.9.
- System disk: at least 100GB; 150GB or more is safer.
- Access: enable both JupyterLab and SSH.
- Persistent storage: mount file storage to `/workspace` if available.

Recommended layout:

```text
/workspace/RISE-master
/workspace/RISE-master/Dataset
/workspace/RISE-master/SINet-V2/snapshot
/workspace/RISE-master/results
```

## 2. Clone Code On Server

```bash
cd /workspace
git clone <YOUR_REPO_URL> RISE-master
cd /workspace/RISE-master
```

For later code updates:

```bash
cd /workspace/RISE-master
git pull
```

## 3. Upload Data Separately

Do not upload datasets through Git. Use one of:

- BitaHub file storage
- BitaHub dataset mounting
- `scp` / `rsync`
- JupyterLab file upload for small archives

The final dataset path should match the training scripts, usually:

```text
/workspace/RISE-master/Dataset
```

## 4. Check GPU And PyTorch

Run these commands after the server starts:

```bash
nvidia-smi
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
df -h
```

## 5. Install Common Dependencies

The selected PyTorch image should already contain PyTorch and CUDA. The fastest
path is to use the setup script included in this repository:

```bash
cd /workspace/RISE-master
bash setup_bitahub.sh
```

This script installs basic tools (`git`, `tmux`, `unzip`), upgrades `pip`, and
installs the Python dependencies listed in
[`requirements_server.txt`](/home/ad/Rise/RISE-master/requirements_server.txt).

If you need to install the packages manually, use:

```bash
pip install opencv-python pillow numpy scipy scikit-image tqdm matplotlib pandas transformers py_sod_metrics
```

If a run reports a missing package, install that package only and rerun.

## 6. Run Long Experiments Safely

Use `tmux` so training continues after the browser or SSH disconnects:

```bash
tmux new -s rise
```

Run the experiment inside the tmux session. Detach with:

```text
Ctrl-b d
```

Reconnect later:

```bash
tmux attach -t rise
```

## 7. Git Rules For This Project

Commit code, scripts, and small documentation only.

Do not commit:

```text
Dataset/
SINet-V2/snapshot/
SINet-V2/pred/
results/
outputs/
output/
logs/
*.pth
*.pt
*.log
*.pid
```

Avoid:

```bash
git add .
```

Prefer explicit staging:

```bash
git add stage3_mask_refine/*.py run_xxx.sh BITAHUB_SERVER_SETUP.md
```
