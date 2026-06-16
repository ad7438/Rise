"""Training CLI for Stage 4 SINet-text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from .common import curriculum_settings, write_json
from .dataset import SINetTextEvalDataset, SINetTextTrainDataset, build_text_collate_fn
from .losses import weighted_boundary_loss, weighted_distillation_loss, weighted_structure_loss
from .model import SINET_ROOT, SINetTextNetwork

if str(SINET_ROOT) not in sys.path:
    sys.path.insert(0, str(SINET_ROOT))

from lib.Network_Res2Net_GRA_NCD import Network  # noqa: E402


def load_tokenizer(text_model_name: str) -> CLIPTokenizer:
    try:
        return CLIPTokenizer.from_pretrained(text_model_name, local_files_only=True)
    except OSError:
        return CLIPTokenizer.from_pretrained(text_model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 4 SINet-text.")
    parser.add_argument("--train_manifest_jsonl", default="Dataset/Stage4CRISLite_baseline/train/manifest.jsonl")
    parser.add_argument("--val_manifest_jsonl", default="Dataset/Stage4CRISLite_baseline/eval/CAMO/manifest.jsonl")
    parser.add_argument("--teacher_path", default="SINet-V2/snapshot/RISE_paper_split/Net_epoch_best.pth")
    parser.add_argument("--student_init", default="SINet-V2/snapshot/RISE_paper_split/Net_epoch_best.pth")
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="./snapshot/SINet_text_v2/")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--text_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_dim", type=int, default=256)
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--disable_text", action="store_true")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--edge_weight", type=float, default=0.3)
    parser.add_argument("--distill_weight", type=float, default=0.2)
    return parser.parse_args()


def configure_device(gpu_id: str) -> torch.device:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def load_gt(gt_path: str) -> np.ndarray:
    gt = np.asarray(Image.open(gt_path).convert("L"), np.float32)
    return gt / (gt.max() + 1e-8)


def load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint


def build_teacher(path: str, device: torch.device) -> Network:
    teacher = Network(channel=32, imagenet_pretrained=False).to(device)
    teacher.load_state_dict(load_state_dict(path))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def validate(model: SINetTextNetwork, data_loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mae_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Stage4 v2 val", leave=False):
            preds = model(
                images=batch["images"].to(device),
                text_input_ids=batch["text_input_ids"].to(device),
                text_attention_mask=batch["text_attention_mask"].to(device),
                category_ids=batch["category_ids"].to(device),
                location_ids=batch["location_ids"].to(device),
                size_ids=batch["size_ids"].to(device),
            )
            probs = torch.sigmoid(preds[3]).cpu().numpy()
            for index, gt_path in enumerate(batch["gt_paths"]):
                if gt_path is None:
                    continue
                gt = load_gt(gt_path)
                pred = probs[index, 0]
                pred = np.array(
                    Image.fromarray((pred * 255).astype(np.uint8)).resize(
                        (gt.shape[1], gt.shape[0]),
                        resample=Image.BILINEAR,
                    ),
                    np.float32,
                )
                pred /= 255.0
                mae_sum += np.abs(pred - gt).mean()
    mae = mae_sum / max(len(data_loader.dataset), 1)
    return {"mae": float(mae)}


def train_one_epoch(
    model: SINetTextNetwork,
    teacher: Network,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[dict, int]:
    model.train()
    totals = {
        "loss": 0.0,
        "structure_init": 0.0,
        "structure_final": 0.0,
        "boundary": 0.0,
        "distill": 0.0,
    }
    steps = 0
    min_confidence, low_scale = curriculum_settings(epoch, args.epoch)

    for step, batch in enumerate(tqdm(data_loader, desc=f"Stage4 v2 train {epoch:03d}", leave=False), start=1):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        base_weights = batch["sample_weights"].to(device)
        final_confidences = batch["final_confidences"].to(device)
        effective_weights = torch.where(
            final_confidences >= min_confidence,
            base_weights,
            base_weights * low_scale,
        )
        if float(effective_weights.sum().item()) <= 1e-8:
            effective_weights = base_weights

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda")
        with amp_context:
            preds = model(
                images=images,
                text_input_ids=batch["text_input_ids"].to(device),
                text_attention_mask=batch["text_attention_mask"].to(device),
                category_ids=batch["category_ids"].to(device),
                location_ids=batch["location_ids"].to(device),
                size_ids=batch["size_ids"].to(device),
            )
            with torch.no_grad():
                teacher_preds = teacher(images)

            structure_init = (
                weighted_structure_loss(preds[0], masks, effective_weights)
                + weighted_structure_loss(preds[1], masks, effective_weights)
                + weighted_structure_loss(preds[2], masks, effective_weights)
            )
            structure_final = weighted_structure_loss(preds[3], masks, effective_weights)
            boundary = weighted_boundary_loss(preds[3], masks, effective_weights)
            distill = weighted_distillation_loss(preds[3], teacher_preds[3], effective_weights)
            loss = structure_init + structure_final + args.edge_weight * boundary + args.distill_weight * distill

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

        global_step += 1
        steps += 1
        totals["loss"] += float(loss.item())
        totals["structure_init"] += float(structure_init.item())
        totals["structure_final"] += float(structure_final.item())
        totals["boundary"] += float(boundary.item())
        totals["distill"] += float(distill.item())

        if step == 1 or step % 20 == 0 or step == len(data_loader):
            logging.info(
                "Epoch [%03d/%03d] Step [%04d/%04d] Loss %.4f Init %.4f Final %.4f Edge %.4f Distill %.4f ConfThr %.2f LowScale %.2f",
                epoch,
                args.epoch,
                step,
                len(data_loader),
                float(loss.item()),
                float(structure_init.item()),
                float(structure_final.item()),
                float(boundary.item()),
                float(distill.item()),
                min_confidence,
                low_scale,
            )
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)

    averages = {key: value / max(steps, 1) for key, value in totals.items()}
    averages["curriculum_min_confidence"] = min_confidence
    averages["curriculum_low_scale"] = low_scale
    for key, value in averages.items():
        writer.add_scalar(f"train/{key}", value, epoch)
    return averages, global_step


def main() -> None:
    args = parse_args()
    device = configure_device(args.gpu_id)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(save_path / "log.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logging.info("SINet-text Train")
    logging.info("Config: %s", vars(args))

    tokenizer = load_tokenizer(args.text_model_name)
    collate_fn = build_text_collate_fn(tokenizer, args.max_text_length)

    train_loader = DataLoader(
        SINetTextTrainDataset(args.train_manifest_jsonl, trainsize=args.trainsize),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SINetTextEvalDataset(args.val_manifest_jsonl, testsize=args.trainsize),
        batch_size=max(1, min(args.batchsize, 8)),
        shuffle=False,
        num_workers=max(1, min(args.workers, 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = SINetTextNetwork(
        channel=32,
        imagenet_pretrained=True,
        text_model_name=args.text_model_name,
        text_dim=args.text_dim,
        freeze_text_encoder=args.freeze_text_encoder,
        use_text=not args.disable_text,
    ).to(device)
    if args.student_init:
        missing, unexpected = model.load_visual_state_dict(load_state_dict(args.student_init))
        logging.info("Student visual init: missing=%d unexpected=%d", len(missing), len(unexpected))

    teacher = build_teacher(args.teacher_path, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epoch))
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None
    writer = SummaryWriter(str(save_path / "summary"))

    best_mae = float("inf")
    best_epoch = 0
    history_path = save_path / "history.jsonl"
    global_step = 0

    for epoch in range(1, args.epoch + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            teacher=teacher,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_structure_init": train_metrics["structure_init"],
            "train_structure_final": train_metrics["structure_final"],
            "train_boundary": train_metrics["boundary"],
            "train_distill": train_metrics["distill"],
            "val_mae": val_metrics["mae"],
            "best_mae": best_mae,
            "best_epoch": best_epoch,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(
            "Epoch: %d, train_loss: %.4f, val_mae: %.6f, bestMAE: %.6f, bestEpoch: %d.",
            epoch,
            train_metrics["loss"],
            val_metrics["mae"],
            best_mae,
            best_epoch,
        )
        print(
            f"Epoch: {epoch}, train_loss: {train_metrics['loss']:.4f}, val_mae: {val_metrics['mae']:.6f}, "
            f"bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}."
        )
        writer.add_scalar("val/mae", val_metrics["mae"], epoch)

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "channel": 32,
                "imagenet_pretrained": False,
                "text_model_name": args.text_model_name,
                "text_dim": args.text_dim,
                "freeze_text_encoder": args.freeze_text_encoder,
                "category_count": 11,
                "location_count": 9,
                "size_count": 3,
                "use_text": not args.disable_text,
            },
        }
        if epoch % 20 == 0:
            torch.save(checkpoint, save_path / f"Net_epoch_{epoch}.pth")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(checkpoint, save_path / "Net_epoch_best.pth")

    writer.close()
    write_json(
        save_path / "train_summary.json",
        {
            "best_epoch": best_epoch,
            "best_mae": best_mae,
            "train_manifest_jsonl": args.train_manifest_jsonl,
            "val_manifest_jsonl": args.val_manifest_jsonl,
            "teacher_path": args.teacher_path,
            "student_init": args.student_init,
            "disable_text": args.disable_text,
        },
    )


if __name__ == "__main__":
    main()
