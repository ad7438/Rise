"""Review helper for Stage 4 eval text outputs."""

from __future__ import annotations

import argparse
import html
import random
import shutil
from pathlib import Path

from .common import load_jsonl_records
from stage2_pseudo_text.mask_utils import (
    create_highlight_overlay,
    crop_with_padding,
    load_binary_mask,
    load_rgb_image,
    preprocess_mask,
    resize_binary_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Stage 4 eval text results into an HTML review page.")
    parser.add_argument("--results_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--high_n", type=int, default=20)
    parser.add_argument("--low_n", type=int, default=20)
    parser.add_argument("--random_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260406)
    return parser.parse_args()


def take_unique(records: list[dict], used: set[str], count: int) -> list[dict]:
    if count <= 0:
        return []
    selected: list[dict] = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in used:
            continue
        used.add(sample_id)
        selected.append(record)
        if len(selected) >= count:
            break
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl_records(Path(args.results_jsonl))
    rng = random.Random(args.seed)

    high_candidates = sorted(
        [record for record in records if not record.get("low_confidence", False)],
        key=lambda record: record.get("final_confidence", 0.0),
        reverse=True,
    )
    low_candidates = sorted(records, key=lambda record: record.get("final_confidence", 0.0))
    random_candidates = list(records)
    rng.shuffle(random_candidates)

    used: set[str] = set()
    selected = []
    selected.extend(take_unique(high_candidates, used, args.high_n))
    selected.extend(take_unique(low_candidates, used, args.low_n))
    selected.extend(take_unique(random_candidates, used, args.random_n))

    rows: list[dict] = []
    for record in selected:
        sample_id = record["sample_id"]
        image = load_rgb_image(record["image_path"])
        mask = load_binary_mask(record["mask_path"])
        mask = resize_binary_mask(mask, image.size)
        clean_mask, metrics = preprocess_mask(mask)
        overlay = create_highlight_overlay(image, clean_mask)
        tight = crop_with_padding(image, metrics.bbox, 0.0)
        context = crop_with_padding(image, metrics.bbox, 0.15)

        overlay_name = f"{sample_id}_overlay.jpg"
        tight_name = f"{sample_id}_tight.jpg"
        context_name = f"{sample_id}_context.jpg"
        mask_name = f"{sample_id}_mask.png"

        overlay.save(assets_dir / overlay_name, quality=95)
        tight.save(assets_dir / tight_name, quality=95)
        context.save(assets_dir / context_name, quality=95)
        shutil.copy2(record["mask_path"], assets_dir / mask_name)

        rows.append(
            {
                "sample_id": sample_id,
                "group": "low_conf" if record.get("low_confidence", False) else "high_conf",
                "final_confidence": float(record.get("final_confidence", 0.0)),
                "category": record.get("category", ""),
                "pseudo_text": record.get("pseudo_text", ""),
                "clip_text": record.get("clip_text", ""),
                "reasons": ", ".join(record.get("low_confidence_reasons") or []),
                "overlay": overlay_name,
                "tight": tight_name,
                "context": context_name,
                "mask": mask_name,
            }
        )

    manifest_path = output_dir / "review_manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("sample_id,group,final_confidence,category,pseudo_text,clip_text,reasons\n")
        for row in rows:
            fields = [
                row["sample_id"],
                row["group"],
                f"{row['final_confidence']:.4f}",
                row["category"].replace(",", " "),
                row["pseudo_text"].replace(",", "，"),
                row["clip_text"].replace(",", " "),
                row["reasons"].replace(",", " "),
            ]
            handle.write(",".join(fields) + "\n")

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Stage4 Eval Text Review</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;font-size:14px;}",
        "th{background:#f5f5f5;position:sticky;top:0;} img{max-width:240px;height:auto;display:block;}",
        ".txt{max-width:280px;line-height:1.5;} .mono{font-family:Consolas,monospace;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(Path(args.results_jsonl).parents[1].name)} 文本抽检样本</h1>",
        (
            f"<p>共 {len(rows)} 张：高置信 {args.high_n}、低置信 {args.low_n}、随机 {args.random_n}。"
            "这里直接复用了已生成的文本结果，只重建可视化，不重跑 Qwen。</p>"
        ),
        "<table>",
        "<thead><tr><th>Group</th><th>ID</th><th>Final</th><th>Category</th><th>中文文本</th><th>English</th><th>Flags</th><th>Overlay</th><th>Tight</th><th>Context</th><th>Mask</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(row['group'])}</td>")
        parts.append(f"<td class=\"mono\">{html.escape(row['sample_id'])}</td>")
        parts.append(f"<td>{row['final_confidence']:.4f}</td>")
        parts.append(f"<td>{html.escape(row['category'])}</td>")
        parts.append(f"<td class=\"txt\">{html.escape(row['pseudo_text'])}</td>")
        parts.append(f"<td class=\"txt\">{html.escape(row['clip_text'])}</td>")
        parts.append(f"<td class=\"txt\">{html.escape(row['reasons'])}</td>")
        parts.append(f"<td><img src=\"assets/{html.escape(row['overlay'])}\"></td>")
        parts.append(f"<td><img src=\"assets/{html.escape(row['tight'])}\"></td>")
        parts.append(f"<td><img src=\"assets/{html.escape(row['context'])}\"></td>")
        parts.append(f"<td><img src=\"assets/{html.escape(row['mask'])}\"></td>")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>", "</body>", "</html>"])

    html_path = output_dir / "review_samples.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    print(html_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
