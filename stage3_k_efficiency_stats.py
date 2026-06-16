#!/usr/bin/env python3
"""Collect K-group efficiency statistics for the final paper pipeline.

This script is intentionally read-mostly. It reuses the completed full-run
logs for authoritative wall-clock timing, runs a small component profile for
offline pseudo-label generation, and benchmarks the final SINet-V2 checkpoint
for training-memory and inference-speed evidence.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "paper_ablation_metrics" / "K_efficiency_stats"
PYTHON = "/home/ad/miniconda3/envs/rise/bin/python"

STAGE2_JSONL = ROOT / "Dataset" / "Stage2PseudoText_full_v4_rich" / "results.jsonl"
REFINE_RESULTS = ROOT / "Dataset" / "Stage3MaskRefine_AGSP_SVPM_SVAC_v1" / "results.jsonl"
GATE_TMP = OUT_DIR / "gate_timing_tmp"

OURS_STAGE1_LOG = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_stage1_train_out.log"
OURS_STAGE2_LOG = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_stage2_train_out.log"
OURS_REFINE_ERR = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_stage3_refine_err.log"
OURS_REFINE_OUT = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_stage3_refine_out.log"
OURS_GATE_OUT = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_gate_out.log"
OURS_BUILD_REHEARSAL_OUT = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_build_rehearsal_out.log"
OURS_TEST_ERR = ROOT / "stage3_v3_edge_agsp_svpm_svac_v1_test_err.log"
OURS_CKPT = ROOT / "SINet-V2" / "snapshot" / "RISE_stage3_v3_edge_agsp_svpm_svac_v1_stage1_auto" / "Net_epoch_best.pth"

RISE_STAGE1_LOG = ROOT / "ablation_m0_only_same_protocol_v1_stage1_train_out.log"
RISE_STAGE2_LOG = ROOT / "ablation_m0_only_same_protocol_v1_stage2_train_out.log"
RISE_TEST_ERR = ROOT / "ablation_m0_only_same_protocol_v1_test_err.log"
RISE_CKPT = ROOT / "SINet-V2" / "snapshot" / "RISE_ablation_m0_only_same_protocol_v1_stage1" / "Net_epoch_best.pth"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_text(command: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip()


def parse_tqdm_total_seconds(path: Path, label: str) -> tuple[int, float] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Example: Stage3 mask refine: 100%|...| 4040/4040 [1:03:59<00:00,  1.05it/s]
    pattern = re.compile(rf"{re.escape(label)}.*?(\d+)/\1\s+\[([0-9:]+)<", re.S)
    matches = pattern.findall(text)
    if not matches:
        # Fallback for ordinary tqdm lines where total appears twice but not as a backref match.
        matches = re.findall(r"(\d+)/(\d+)\s+\[([0-9:]+)<", text)
        if not matches:
            return None
        total = int(matches[-1][1])
        elapsed_text = matches[-1][2]
    else:
        total = int(matches[-1][0])
        elapsed_text = matches[-1][1]
    parts = [int(part) for part in elapsed_text.split(":")]
    if len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    else:
        seconds = float(parts[0])
    return total, float(seconds)


def parse_first_last_timestamps(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    timestamp_re = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
    timestamps = [datetime.fromisoformat(match.group(1)) for match in timestamp_re.finditer(text)]
    epoch_matches = re.findall(r"Epoch:\s*(\d+),\s*joint_four_metrics\s*([0-9.]+).*?bestEpoch:\s*(\d+)", text)
    step_match = re.search(r"Step \[\d+/(\d+)\]", text)
    first = timestamps[0] if timestamps else None
    last = timestamps[-1] if timestamps else None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return {
        "first_timestamp": first.isoformat(sep=" ") if first else "",
        "last_step_timestamp": last.isoformat(sep=" ") if last else "",
        "file_mtime": mtime.isoformat(sep=" "),
        "step_logged_seconds": (last - first).total_seconds() if first and last else "",
        "wall_seconds_to_file_mtime": (mtime - first).total_seconds() if first else "",
        "epochs_logged": int(epoch_matches[-1][0]) if epoch_matches else "",
        "best_epoch": int(epoch_matches[-1][2]) if epoch_matches else "",
        "final_joint_metric": float(epoch_matches[-1][1]) if epoch_matches else "",
        "steps_per_epoch": int(step_match.group(1)) if step_match else "",
    }


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def profile_offline_components(limit: int = 40, warmup: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image

    from stage3_mask_refine.agsp import build_agsp_prior
    from stage3_mask_refine.common import load_image_and_clean_mask, load_stage2_records
    from stage3_mask_refine.fuse import is_edge_preserving_target
    from stage3_mask_refine.pipeline_cli import _svpm_candidate_mask, _text_for_record
    from stage3_mask_refine.svac import build_svac_refined_mask
    from stage3_mask_refine.svpm import build_svpm_prior_with_debug
    from stage3_mask_refine.text_prior import CLIPSegTextPrior
    from stage3_mask_refine.visual_refine import adaptive_radii

    records = load_stage2_records(STAGE2_JSONL, limit=limit)
    t0 = time.perf_counter()
    text_prior_model = CLIPSegTextPrior(model_name="CIDAS/clipseg-rd64-refined", device="cuda", hf_endpoint=None)
    cuda_sync()
    model_load_seconds = time.perf_counter() - t0

    measured_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = defaultdict(list)

    for index, record in enumerate(records):
        measured = index >= warmup
        stage_times: dict[str, float] = {}

        t = time.perf_counter()
        image, init_mask, metrics = load_image_and_clean_mask(
            record,
            min_component_area_pixels=64,
            min_component_area_ratio=0.0005,
        )
        image_np = np.array(image)
        cuda_sync()
        stage_times["m0_image_mask_load_clean"] = time.perf_counter() - t

        if metrics.is_empty:
            continue

        t = time.perf_counter()
        text = _text_for_record(record)
        ps_raw = text_prior_model.predict(image, text)
        cuda_sync()
        stage_times["text_lookup_clipseg_response"] = time.perf_counter() - t

        t = time.perf_counter()
        text_prior, agsp_anchor, agsp_mf0 = build_agsp_prior(
            ps_raw=ps_raw,
            m0=init_mask,
            anchor_radius=25,
            anchor_blur=7,
            mask_blur=5,
            lambda_s=0.2,
            semantic_prior_mode="agsp_full",
        )
        text_prior = np.asarray(text_prior, dtype=np.float32)
        agsp_mf0 = np.asarray(agsp_mf0, dtype=np.float32)
        stage_times["agsp_prior"] = time.perf_counter() - t

        t = time.perf_counter()
        _, _, _ = adaptive_radii(init_mask)
        _ = is_edge_preserving_target(init_mask, str(record.get("category") or "unknown"))
        vis_soft, _, _ = build_svpm_prior_with_debug(
            image=image_np,
            m0=init_mask,
            ps_agsp=text_prior,
            n_segments=300,
            compactness=10.0,
            dilate_radius=25,
            alpha=0.6,
            beta=0.4,
            blur_ksize=5,
            visual_prior_mode="svpm_full",
        )
        vis_soft = np.asarray(vis_soft, dtype=np.float32)
        _ = _svpm_candidate_mask(vis_soft, init_mask)
        stage_times["svpm_visual_prior"] = time.perf_counter() - t

        t = time.perf_counter()
        refined_mask, _, _, _ = build_svac_refined_mask(
            m0=init_mask,
            ps_agsp=text_prior,
            pv=vis_soft,
            semantic_confidence=float(record.get("final_confidence") or 0.0),
            mf0=agsp_mf0,
            base_radius=10,
            expand_radius=35,
            local_radius=7,
            visual_threshold=0.5,
            component_threshold=0.45,
            binarize_threshold=0.5,
            alpha_o=0.50,
            alpha_s=0.35,
            alpha_d=0.15,
            high_conf=0.75,
            mid_conf=0.55,
            high_weights=(0.25, 0.30, 0.45),
            mid_weights=(0.35, 0.35, 0.30),
            low_weights=(0.50, 0.35, 0.15),
            score_mode="weighted_sum",
            fusion_mode="tiered",
            use_anchor_score=True,
            use_semantic_score=True,
            use_spatial_score=True,
        )
        stage_times["svac_refinement"] = time.perf_counter() - t

        t = time.perf_counter()
        _ = Image.fromarray((refined_mask > 0).astype(np.uint8) * 255)
        stage_times["mask_encode_no_disk_write"] = time.perf_counter() - t

        if measured:
            row = {"sample_id": record["sample_id"], "profile_index": index, **stage_times}
            row["total_profiled_seconds"] = sum(stage_times.values())
            measured_rows.append(row)
            for key, value in stage_times.items():
                buckets[key].append(value)
            buckets["total_profiled_seconds"].append(row["total_profiled_seconds"])

    summary = {
        "limit": limit,
        "warmup_samples": warmup,
        "measured_samples": len(measured_rows),
        "clipseg_model_load_seconds": model_load_seconds,
        "component_means_seconds": {key: mean(values) for key, values in buckets.items()},
        "component_std_seconds": {key: stdev(values) if len(values) > 1 else 0.0 for key, values in buckets.items()},
    }
    return measured_rows, summary


def run_gate_timing() -> dict[str, Any]:
    if GATE_TMP.exists():
        import shutil

        shutil.rmtree(GATE_TMP)
    cmd = [
        PYTHON,
        str(ROOT / "stage3_auto_gate_refined.py"),
        "--results_jsonl",
        str(REFINE_RESULTS.relative_to(ROOT)),
        "--output_root",
        str(GATE_TMP.relative_to(ROOT)),
        "--no-save_gate_debug",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(result.stdout + "\n" + result.stderr)
    summary_path = GATE_TMP / "auto_gate_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "gate_command": " ".join(cmd),
        "gate_wall_seconds": elapsed,
        "train_pool_total": summary.get("train_pool_total", 0),
        "decision_counts": summary.get("decision_counts", {}),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": run_text(["hostname"]),
        "python": sys.version,
        "nvidia_smi": run_text(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || true"]),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_runtime_version"] = torch.version.cuda
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def load_sinet_model(checkpoint: Path, train_mode: bool) -> Any:
    import torch

    sinet_root = ROOT / "SINet-V2"
    sys.path.insert(0, str(sinet_root))
    from lib.Network_Res2Net_GRA_NCD import Network

    model = Network(imagenet_pretrained=False)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.cuda()
    if train_mode:
        model.train()
    else:
        model.eval()
    return model


def benchmark_training_memory(checkpoint: Path, batch_size: int = 16, image_size: int = 352) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    model = load_sinet_model(checkpoint, train_mode=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        outputs = model(x)
        loss = sum(output.mean() for output in outputs)
        loss.backward()
        optimizer.step()

    for _ in range(2):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    result = {
        "cuda_available": True,
        "batch_size": batch_size,
        "image_size": image_size,
        "single_step_seconds": elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }
    del model
    torch.cuda.empty_cache()
    return result


def benchmark_inference(checkpoint: Path, warmup: int = 30, iterations: int = 300, image_size: int = 352) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    model = load_sinet_model(checkpoint, train_mode=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    x = torch.randn(1, 3, image_size, image_size, device="cuda")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    result = {
        "cuda_available": True,
        "warmup": warmup,
        "iterations": iterations,
        "batch_size": 1,
        "image_size": image_size,
        "total_seconds": elapsed,
        "latency_ms": elapsed / iterations * 1000.0,
        "fps": iterations / elapsed,
        "params": params,
        "params_million": params / 1_000_000.0,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "text_model_required_at_inference": False,
    }
    try:
        import thop  # type: ignore

        macs, thop_params = thop.profile(model, inputs=(x,), verbose=False)
        result["macs"] = int(macs)
        result["gmacs"] = macs / 1_000_000_000.0
        result["thop_params"] = int(thop_params)
    except Exception as exc:
        result["flops_note"] = f"THOP unavailable or failed: {exc!r}"
    del model
    torch.cuda.empty_cache()
    return result


def build_offline_rows(component_summary: dict[str, Any], gate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    full_refine = parse_tqdm_total_seconds(OURS_REFINE_ERR, "Stage3 mask refine")
    if full_refine is None:
        total_samples = 4040
        full_refine_seconds = ""
    else:
        total_samples, full_refine_seconds = full_refine
    train_pool = int(gate_summary["train_pool_total"])
    component_means = component_summary["component_means_seconds"]
    component_stds = component_summary["component_std_seconds"]

    rows: list[dict[str, Any]] = []
    for stage, note in [
        ("m0_image_mask_load_clean", "Read image and precomputed M0, then clean tiny components."),
        ("text_lookup_clipseg_response", "Use existing text description and run CLIPSeg text response."),
        ("agsp_prior", "Anchor-guided semantic prior."),
        ("svpm_visual_prior", "Superpixel-guided visual prior."),
        ("svac_refinement", "Semantic-visual anchor-constrained component refinement."),
        ("mask_encode_no_disk_write", "Mask encoding overhead measured without disk write."),
    ]:
        mean_seconds = float(component_means.get(stage, 0.0))
        rows.append(
            {
                "stage": stage,
                "samples": component_summary["measured_samples"],
                "mean_seconds_per_sample": f"{mean_seconds:.6f}",
                "std_seconds_per_sample": f"{float(component_stds.get(stage, 0.0)):.6f}",
                "projected_total_seconds_for_4040": f"{mean_seconds * total_samples:.2f}" if full_refine else "",
                "projected_total_minutes_for_4040": f"{mean_seconds * total_samples / 60.0:.2f}" if full_refine else "",
                "evidence": "20 measured samples after 20 warm-up samples",
                "notes": note,
            }
        )
    rows.append(
        {
            "stage": "full_stage3_refine_actual",
            "samples": total_samples,
            "mean_seconds_per_sample": f"{full_refine_seconds / total_samples:.6f}" if full_refine else "",
            "std_seconds_per_sample": "",
            "projected_total_seconds_for_4040": f"{full_refine_seconds:.2f}" if full_refine else "",
            "projected_total_minutes_for_4040": f"{full_refine_seconds / 60.0:.2f}" if full_refine else "",
            "evidence": str(OURS_REFINE_ERR.relative_to(ROOT)),
            "notes": "Authoritative completed full-run wall-clock from tqdm; includes disk writes and visual asset saving.",
        }
    )
    rows.append(
        {
            "stage": "auto_gate",
            "samples": train_pool,
            "mean_seconds_per_sample": f"{gate_summary['gate_wall_seconds'] / max(train_pool, 1):.6f}",
            "std_seconds_per_sample": "",
            "projected_total_seconds_for_4040": f"{gate_summary['gate_wall_seconds']:.2f}",
            "projected_total_minutes_for_4040": f"{gate_summary['gate_wall_seconds'] / 60.0:.2f}",
            "evidence": "fresh timing run without dataset/debug-image build",
            "notes": "Decision-only gate timing; final dataset symlink creation is filesystem dependent and negligible relative to CLIPSeg.",
        }
    )
    return rows


def build_training_rows(training_memory: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = [
        ("K2-RISE-local", "RISE local M0-only", RISE_STAGE1_LOG, RISE_STAGE2_LOG, RISE_CKPT),
        ("K2-Ours", "AGSP+SVPM+SVAC v1", OURS_STAGE1_LOG, OURS_STAGE2_LOG, OURS_CKPT),
    ]
    rows: list[dict[str, Any]] = []
    for exp_id, name, stage1_log, stage2_log, checkpoint in experiments:
        for stage_name, log_path, epochs, lr in [
            ("stage1", stage1_log, 100, "1e-4"),
            ("stage2", stage2_log, 40, "5e-5"),
        ]:
            parsed = parse_first_last_timestamps(log_path)
            rows.append(
                {
                    "id": exp_id,
                    "method": name,
                    "stage": stage_name,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": 16,
                    "steps_per_epoch": parsed["steps_per_epoch"],
                    "first_timestamp": parsed["first_timestamp"],
                    "last_step_timestamp": parsed["last_step_timestamp"],
                    "file_mtime": parsed["file_mtime"],
                    "wall_seconds_to_file_mtime": f"{float(parsed['wall_seconds_to_file_mtime']):.2f}",
                    "wall_minutes_to_file_mtime": f"{float(parsed['wall_seconds_to_file_mtime']) / 60.0:.2f}",
                    "best_epoch": parsed["best_epoch"],
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "peak_allocated_mb_representative_batch": f"{training_memory.get('peak_allocated_mb', ''):.2f}" if training_memory.get("cuda_available") else "",
                    "peak_reserved_mb_representative_batch": f"{training_memory.get('peak_reserved_mb', ''):.2f}" if training_memory.get("cuda_available") else "",
                    "memory_note": "Same SINet-V2 training graph for RISE local and Ours; measured on one representative batch size 16.",
                }
            )
    return rows


def build_inference_rows(inference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "K3",
            "method": "Final COD model / SINet-V2",
            "checkpoint": str(OURS_CKPT.relative_to(ROOT)),
            "input_size": inference.get("image_size", 352),
            "batch_size": inference.get("batch_size", 1),
            "params_million": f"{float(inference.get('params_million', 0.0)):.3f}" if inference.get("cuda_available") else "",
            "latency_ms": f"{float(inference.get('latency_ms', 0.0)):.3f}" if inference.get("cuda_available") else "",
            "fps": f"{float(inference.get('fps', 0.0)):.3f}" if inference.get("cuda_available") else "",
            "peak_allocated_mb": f"{float(inference.get('peak_allocated_mb', 0.0)):.2f}" if inference.get("cuda_available") else "",
            "peak_reserved_mb": f"{float(inference.get('peak_reserved_mb', 0.0)):.2f}" if inference.get("cuda_available") else "",
            "gmacs": f"{float(inference.get('gmacs', 0.0)):.3f}" if "gmacs" in inference else "",
            "text_model_required_at_inference": inference.get("text_model_required_at_inference", False),
            "benchmark": f"{inference.get('warmup', '')} warm-up + {inference.get('iterations', '')} timed random-input iterations on GPU",
            "notes": inference.get("flops_note", ""),
        }
    ]


def write_notes(
    offline_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> None:
    full_row = next(row for row in offline_rows if row["stage"] == "full_stage3_refine_actual")
    gate_row = next(row for row in offline_rows if row["stage"] == "auto_gate")
    ours_stage1 = next(row for row in training_rows if row["id"] == "K2-Ours" and row["stage"] == "stage1")
    inference = inference_rows[0]
    text = f"""# K efficiency statistics

Purpose: record efficiency evidence for the final AGSP+SVPM+SVAC v1 paper pipeline.

K1 offline pseudo-label generation:
- Full Stage-3 refinement run: {full_row['projected_total_minutes_for_4040']} min for {full_row['samples']} samples ({full_row['mean_seconds_per_sample']} s/sample).
- Auto gate decision-only run: {gate_row['projected_total_seconds_for_4040']} s for {gate_row['samples']} train-pool samples.
- Component breakdown is measured on 20 samples after 20 warm-up samples. Use it as relative cost attribution; the completed full-run tqdm is the authoritative total runtime.

K2 training:
- Ours Stage-1 wall time from log/file mtime: {ours_stage1['wall_minutes_to_file_mtime']} min, 100 epochs, batch size 16.
- RISE local and Ours use the same SINet-V2 training graph; representative batch peak memory is recorded once and shared.

K3 inference:
- Final COD model uses no text model at inference.
- Params: {inference['params_million']}M.
- Latency: {inference['latency_ms']} ms/image; FPS: {inference['fps']}.

Hardware:
```json
{json.dumps(hardware, ensure_ascii=False, indent=2)}
```

Limitations:
- K1 component attribution is profiled on a small subset to avoid re-running the one-hour CLIPSeg pipeline.
- K2 peak memory is a representative forward/backward batch measurement, not a continuously sampled full-training maximum.
"""
    (OUT_DIR / "notes.md").write_text(text, encoding="utf-8")


def write_git_state() -> None:
    status = run_text(["git", "status", "--short", "--branch"])
    log = run_text(["git", "log", "--oneline", "-8"])
    (OUT_DIR / "git_state.txt").write_text(log + "\n\n" + status + "\n", encoding="utf-8")


def write_command() -> None:
    (OUT_DIR / "command.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/ad/Rise/RISE-master\n"
        "/home/ad/miniconda3/envs/rise/bin/python stage3_k_efficiency_stats.py\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    hardware = hardware_info()
    write_json(OUT_DIR / "hardware_info.json", hardware)

    profile_rows, profile_summary = profile_offline_components(limit=40, warmup=20)
    write_csv(OUT_DIR / "k1_component_profile_samples.csv", profile_rows)
    write_json(OUT_DIR / "k1_component_profile_summary.json", profile_summary)

    gate_summary = run_gate_timing()
    write_json(OUT_DIR / "k1_gate_timing.json", gate_summary)

    offline_rows = build_offline_rows(profile_summary, gate_summary)
    write_csv(OUT_DIR / "efficiency_offline.csv", offline_rows)

    training_memory = benchmark_training_memory(OURS_CKPT, batch_size=16, image_size=352)
    write_json(OUT_DIR / "k2_training_memory.json", training_memory)
    training_rows = build_training_rows(training_memory)
    write_csv(OUT_DIR / "efficiency_training.csv", training_rows)

    inference = benchmark_inference(OURS_CKPT, warmup=30, iterations=300, image_size=352)
    write_json(OUT_DIR / "k3_inference_benchmark.json", inference)
    inference_rows = build_inference_rows(inference)
    write_csv(OUT_DIR / "efficiency_inference.csv", inference_rows)

    write_notes(offline_rows, training_rows, inference_rows, hardware)
    (OUT_DIR / "config.yaml").write_text(
        "\n".join(
            [
                "experiment: K_efficiency_stats",
                "branch: codex/k-efficiency-stats",
                "main_method: AGSP+SVPM+SVAC v1 Stage1-only",
                "offline_profile: 20 warm-up samples + 20 measured samples",
                "full_refine_timing_source: stage3_v3_edge_agsp_svpm_svac_v1_stage3_refine_err.log",
                "training_timing_source: train logs + file mtimes",
                "inference_checkpoint: SINet-V2/snapshot/RISE_stage3_v3_edge_agsp_svpm_svac_v1_stage1_auto/Net_epoch_best.pth",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_command()
    write_git_state()
    print(f"K efficiency artifacts written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
