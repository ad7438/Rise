"""Training CLI for Stage 4 CRIS-lite."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from vis import EvaluationMetricsV2

from .dataset import CRISLiteEvalDataset, CRISLiteTrainDataset, build_text_collate_fn
from .losses import weighted_structure_loss
from .model import CRISLiteModel


def load_tokenizer(text_model_name: str) -> CLIPTokenizer:
    try:
        return CLIPTokenizer.from_pretrained(text_model_name, local_files_only=True)
    except OSError:
        return CLIPTokenizer.from_pretrained(text_model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 4 CRIS-lite.")
    parser.add_argument("--train_manifest_jsonl", default="Dataset/Stage4CRISLite_baseline/train/manifest.jsonl")
    parser.add_argument("--val_manifest_jsonl", default="Dataset/Stage4CRISLite_baseline/eval/CAMO/manifest.jsonl")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="./snapshot/CRIS_lite_baseline/")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--backbone_name", type=str, default="resnet50")
    parser.add_argument("--text_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--freeze_text_encoder", dest="freeze_text_encoder", action="store_true")
    parser.add_argument("--no_freeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    parser.set_defaults(freeze_text_encoder=True)
    parser.add_argument("--backbone_pretrained", dest="backbone_pretrained", action="store_true")
    parser.add_argument("--no_backbone_pretrained", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=True)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
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


def validate(model: CRISLiteModel, data_loader: DataLoader, device: torch.device) -> dict:
    metric = EvaluationMetricsV2()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Stage4 val", leave=False):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(images, input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

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
                metric.step(pred=pred, gt=gt)

    metric_dic = metric.get_results()
    return {
        "sm": float(metric_dic["sm"]),
        "emMean": float(metric_dic["emMean"]),
        "wfm": float(metric_dic["wfm"]),
        "mae": float(metric_dic["mae"]),
    }


def train_one_epoch(
    model: CRISLiteModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    epoch_loss = 0.0
    steps = 0

    for step, batch in enumerate(tqdm(data_loader, desc=f"Stage4 train {epoch:03d}", leave=False), start=1):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sample_weights = batch["sample_weights"].to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast(device_type=device.type, enabled=use_amp)
        with amp_context:
            logits = model(images, input_ids, attention_mask)
            loss = weighted_structure_loss(logits, masks, sample_weights)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradient(optimizer, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_gradient(optimizer, grad_clip)
            optimizer.step()

        global_step += 1
        steps += 1
        epoch_loss += float(loss.item())

        if step == 1 or step % 20 == 0 or step == len(data_loader):
            logging.info(
                "Epoch [%03d/%03d] Step [%04d/%04d] Loss %.4f",
                epoch,
                total_epochs,
                step,
                len(data_loader),
                float(loss.item()),
            )
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)

    avg_loss = epoch_loss / max(steps, 1)
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    return avg_loss, global_step


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
    logging.info("CRIS-lite Train")
    logging.info("Config: %s", vars(args))

    tokenizer = load_tokenizer(args.text_model_name)
    collate_fn = build_text_collate_fn(tokenizer, args.max_text_length)

    train_loader = DataLoader(
        CRISLiteTrainDataset(args.train_manifest_jsonl, trainsize=args.trainsize),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        CRISLiteEvalDataset(args.val_manifest_jsonl, testsize=args.trainsize),
        batch_size=max(1, min(args.batchsize, 8)),
        shuffle=False,
        num_workers=max(1, min(args.workers, 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = CRISLiteModel(
        backbone_name=args.backbone_name,
        backbone_pretrained=args.backbone_pretrained,
        text_model_name=args.text_model_name,
        decoder_dim=args.decoder_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_text_encoder=args.freeze_text_encoder,
    ).to(device)

    if args.load:
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epoch - 1))
    scaler = torch.amp.GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
    writer = SummaryWriter(str(save_path / "summary"))

    best_mae = float("inf")
    best_epoch = 0
    history_path = save_path / "history.jsonl"
    global_step = 0

    for epoch in range(1, args.epoch + 1):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.use_amp and device.type == "cuda",
            args.clip,
            epoch,
            args.epoch,
            writer,
            global_step,
        )
        metrics = validate(model, val_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": metrics["mae"],
            "val_sm": metrics["sm"],
            "val_emMean": metrics["emMean"],
            "val_wfm": metrics["wfm"],
            "lr": optimizer.param_groups[0]["lr"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        writer.add_scalar("val/mae", metrics["mae"], epoch)
        writer.add_scalar("val/sm", metrics["sm"], epoch)
        writer.add_scalar("val/emMean", metrics["emMean"], epoch)
        writer.add_scalar("val/wfm", metrics["wfm"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "backbone_name": args.backbone_name,
                "backbone_pretrained": False,
                "text_model_name": args.text_model_name,
                "decoder_dim": args.decoder_dim,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "freeze_text_encoder": args.freeze_text_encoder,
            },
            "epoch": epoch,
            "metrics": metrics,
        }

        if epoch % 25 == 0:
            torch.save(checkpoint, save_path / f"Net_epoch_{epoch}.pth")

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch
            torch.save(checkpoint, save_path / "Net_epoch_best.pth")
            logging.info("Saved best checkpoint at epoch %03d", epoch)

        logging.info(
            "Epoch %03d train_loss %.4f val_mae %.6f best_mae %.6f best_epoch %d",
            epoch,
            train_loss,
            metrics["mae"],
            best_mae,
            best_epoch,
        )
        print(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, val_mae: {metrics['mae']:.6f}, "
            f"bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}."
        )

    with (save_path / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_mae": best_mae,
                "train_manifest_jsonl": args.train_manifest_jsonl,
                "val_manifest_jsonl": args.val_manifest_jsonl,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
