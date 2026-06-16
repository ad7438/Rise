#!/usr/bin/env python3
"""Export paginated full-review HTML for Stage 3 results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_SELECTION_KEY = "stage3-mask-review-selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full Stage 3 review pages.")
    parser.add_argument(
        "--results_jsonl",
        default="Dataset/Stage3MaskRefine_v3_edge/results.jsonl",
        help="Stage 3 results JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        default="Dataset/Stage3MaskRefine_v3_edge/review_full_select",
        help="Output directory for review HTML.",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=100,
        help="Samples per page.",
    )
    parser.add_argument(
        "--sort_by",
        default="change_ratio",
        choices=["change_ratio", "final_confidence", "sample_id"],
        help="Sort key.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending.",
    )
    parser.add_argument(
        "--title",
        default="Stage3 全量审阅",
        help="Page title shown in generated HTML.",
    )
    parser.add_argument(
        "--selection_key",
        default=DEFAULT_SELECTION_KEY,
        help="Browser localStorage key for checkbox selections.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def maybe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def relpath_posix(target: Path, start: Path) -> str:
    return Path(os.path.relpath(str(target), str(start))).as_posix()


def page_nav(page_number: int, total_pages: int) -> str:
    parts: list[str] = []
    if page_number > 1:
        parts.append(f"<a href='page_{page_number - 1:04d}.html'>上一页</a>")
    if page_number < total_pages:
        parts.append(f"<a href='page_{page_number + 1:04d}.html'>下一页</a>")
    parts.append("<a href='index.html'>目录</a>")
    return " | ".join(parts)


def build_rows(records: list[dict[str, Any]], page_dir: Path, repo_root: Path) -> str:
    rows: list[str] = []
    for record in records:
        sample_id = record["sample_id"]
        image_path = (repo_root / record["image_path"]).resolve()
        old_mask_path = (repo_root / record["init_mask_path"]).resolve()
        refined_mask_path = (repo_root / record["refined_mask_path"]).resolve()
        gt_path = (repo_root / "Dataset/TrainDataset/GT" / f"{sample_id}.png").resolve()

        image_rel = relpath_posix(image_path, page_dir)
        old_mask_rel = relpath_posix(old_mask_path, page_dir)
        refined_mask_rel = relpath_posix(refined_mask_path, page_dir)
        gt_rel = relpath_posix(gt_path, page_dir) if gt_path.exists() else ""

        training_text = record.get("training_text") or record.get("clip_text") or ""
        gt_cell = f"<img loading='lazy' src='{html.escape(gt_rel)}' />" if gt_rel else ""
        rows.append(
            "<tr>"
            f"<td class='sel'><input type='checkbox' class='pick-box' data-id='{html.escape(sample_id)}' /></td>"
            f"<td>{html.escape(sample_id)}</td>"
            f"<td>{maybe_float(record.get('change_ratio')):.4f}</td>"
            f"<td>{maybe_float(record.get('final_confidence')):.4f}</td>"
            f"<td>{html.escape(record.get('category', ''))}</td>"
            f"<td>{html.escape(record.get('refine_mode', ''))}</td>"
            f"<td>{html.escape(record.get('refine_submode', ''))}</td>"
            f"<td>{'yes' if record.get('low_confidence') else 'no'}</td>"
            f"<td>{'yes' if record.get('dropped_from_stage4') else 'no'}</td>"
            f"<td class='text-cell'>{html.escape(training_text)}</td>"
            f"<td><img loading='lazy' src='{html.escape(image_rel)}' /></td>"
            f"<td><canvas class='overlay-canvas' data-image='{html.escape(image_rel)}' data-mask='{html.escape(old_mask_rel)}'></canvas></td>"
            f"<td><canvas class='overlay-canvas' data-image='{html.escape(image_rel)}' data-mask='{html.escape(refined_mask_rel)}'></canvas></td>"
            f"<td><img loading='lazy' src='{html.escape(old_mask_rel)}' /></td>"
            f"<td><img loading='lazy' src='{html.escape(refined_mask_rel)}' /></td>"
            f"<td>{gt_cell}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_assets(output_dir: Path, selection_key: str) -> None:
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    css_text = """
body{font-family:Arial,sans-serif;margin:20px;}
table{border-collapse:collapse;width:100%;table-layout:fixed;}
th,td{border:1px solid #ccc;padding:6px;vertical-align:top;word-break:break-word;}
th{position:sticky;top:0;background:#fff;z-index:2;}
img,canvas{width:220px;height:165px;max-width:220px;max-height:165px;object-fit:contain;background:#111;}
.toolbar{position:sticky;top:0;background:#fff;padding:8px 0;z-index:3;}
.text-cell{min-width:260px;max-width:360px;}
.sel{width:36px;text-align:center;}
.page-list{columns:4;}
"""
    js_text = f"""
const STORAGE_KEY = "{selection_key}";

function loadSelection() {{
  try {{
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
  }} catch (err) {{
    return {{}};
  }}
}}

function saveSelection(selection) {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
}}

function updateSelectedCount() {{
  const selected = loadSelection();
  const count = Object.values(selected).filter(Boolean).length;
  document.querySelectorAll("#selected-count").forEach((node) => {{
    node.textContent = `已选 ${{count}}`;
  }});
}}

function initCheckboxes() {{
  const selected = loadSelection();
  document.querySelectorAll(".pick-box").forEach((box) => {{
    const sampleId = box.dataset.id;
    box.checked = Boolean(selected[sampleId]);
    box.addEventListener("change", () => {{
      const next = loadSelection();
      if (box.checked) {{
        next[sampleId] = true;
      }} else {{
        delete next[sampleId];
      }}
      saveSelection(next);
      updateSelectedCount();
    }});
  }});
  updateSelectedCount();
}}

function selectVisible(flag) {{
  const next = loadSelection();
  document.querySelectorAll(".pick-box").forEach((box) => {{
    box.checked = flag;
    if (flag) {{
      next[box.dataset.id] = true;
    }} else {{
      delete next[box.dataset.id];
    }}
  }});
  saveSelection(next);
  updateSelectedCount();
}}

function clearSelection() {{
  localStorage.removeItem(STORAGE_KEY);
  document.querySelectorAll(".pick-box").forEach((box) => {{
    box.checked = false;
  }});
  updateSelectedCount();
}}

function exportSelection() {{
  const selected = loadSelection();
  const ids = Object.keys(selected).filter((id) => selected[id]).sort();
  const lines = ["sample_id,keep"];
  ids.forEach((id) => lines.push(`${{id}},1`));
  const blob = new Blob([lines.join("\\n")], {{type: "text/csv;charset=utf-8;"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "stage3_selected_samples.csv";
  link.click();
  URL.revokeObjectURL(url);
}}

function drawOverlay(canvas) {{
  const imageSrc = canvas.dataset.image;
  const maskSrc = canvas.dataset.mask;
  const ctx = canvas.getContext("2d");
  const image = new Image();
  const mask = new Image();
  let ready = 0;

  function maybeRender() {{
    ready += 1;
    if (ready < 2) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);

    const temp = document.createElement("canvas");
    temp.width = image.naturalWidth;
    temp.height = image.naturalHeight;
    const tctx = temp.getContext("2d");
    tctx.drawImage(mask, 0, 0, temp.width, temp.height);
    const maskData = tctx.getImageData(0, 0, temp.width, temp.height);
    const overlay = ctx.createImageData(temp.width, temp.height);

    for (let i = 0; i < maskData.data.length; i += 4) {{
      const alpha = maskData.data[i] > 0 ? 110 : 0;
      overlay.data[i] = 255;
      overlay.data[i + 1] = 40;
      overlay.data[i + 2] = 40;
      overlay.data[i + 3] = alpha;
    }}

    ctx.putImageData(overlay, 0, 0);
  }}

  image.onload = maybeRender;
  mask.onload = maybeRender;
  image.src = imageSrc;
  mask.src = maskSrc;
}}

function initOverlays() {{
  document.querySelectorAll(".overlay-canvas").forEach((canvas) => drawOverlay(canvas));
}}

window.addEventListener("DOMContentLoaded", () => {{
  initCheckboxes();
  initOverlays();
}});
"""
    (assets_dir / "review.css").write_text(css_text.strip() + "\n", encoding="utf-8")
    (assets_dir / "review.js").write_text(js_text.strip() + "\n", encoding="utf-8")


def write_selection_template(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_path = output_dir / "selection_template.csv"
    fieldnames = [
        "sample_id",
        "keep",
        "notes",
        "change_ratio",
        "final_confidence",
        "category",
        "refine_mode",
        "refine_submode",
        "low_confidence",
        "dropped_from_stage4",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "keep": "",
                    "notes": "",
                    "change_ratio": f"{maybe_float(record.get('change_ratio')):.6f}",
                    "final_confidence": f"{maybe_float(record.get('final_confidence')):.6f}",
                    "category": record.get("category", ""),
                    "refine_mode": record.get("refine_mode", ""),
                    "refine_submode": record.get("refine_submode", ""),
                    "low_confidence": str(bool(record.get("low_confidence", False))).lower(),
                    "dropped_from_stage4": str(bool(record.get("dropped_from_stage4", False))).lower(),
                }
            )


def write_index(
    output_dir: Path,
    total_samples: int,
    total_pages: int,
    page_size: int,
    summary: dict[str, Any] | None,
    title: str,
) -> None:
    links = "\n".join(
        f"<li><a href='page_{page_no:04d}.html'>第 {page_no} 页</a></li>"
        for page_no in range(1, total_pages + 1)
    )
    summary_html = ""
    if summary:
        summary_html = (
            "<ul>"
            f"<li>总样本: {summary.get('total_samples', total_samples)}</li>"
            f"<li>平均变化: {maybe_float(summary.get('average_change_ratio')):.4f}</li>"
            f"<li>中位变化: {maybe_float(summary.get('median_change_ratio')):.4f}</li>"
            f"<li>大于 1% 变化: {summary.get('samples_above_1pct_change', '')}</li>"
            f"<li>大于 2% 变化: {summary.get('samples_above_2pct_change', '')}</li>"
            "</ul>"
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="assets/review.css" />
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>总样本 {total_samples}，每页 {page_size} 张，共 {total_pages} 页。</p>
  {summary_html}
  <div class="toolbar">
    <button onclick="clearSelection()">清空已选</button>
    <button onclick="exportSelection()">导出已选 CSV</button>
    <span id="selected-count">已选 0</span>
  </div>
  <p>你可以直接勾选。勾选结果保存在浏览器本地。导出后把 CSV 给我，我就能按它构建训练子集。</p>
  <ol class="page-list">
    {links}
  </ol>
  <script src="assets/review.js"></script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_page(
    output_dir: Path,
    page_number: int,
    total_pages: int,
    records: list[dict[str, Any]],
    page_size: int,
    repo_root: Path,
    title: str,
) -> None:
    page_dir = output_dir
    rows = build_rows(records, page_dir, repo_root)
    start_idx = (page_number - 1) * page_size + 1
    end_idx = start_idx + len(records) - 1
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)} Page {page_number}</title>
  <link rel="stylesheet" href="assets/review.css" />
</head>
<body>
  <h1>{html.escape(title)} 第 {page_number}/{total_pages} 页</h1>
  <p>{page_nav(page_number, total_pages)}</p>
  <div class="toolbar">
    <button onclick="selectVisible(true)">本页全选</button>
    <button onclick="selectVisible(false)">本页清空</button>
    <button onclick="exportSelection()">导出已选 CSV</button>
    <span id="selected-count">已选 0</span>
  </div>
  <p>当前页样本 {start_idx} - {end_idx}</p>
  <table>
    <thead>
      <tr>
        <th>选</th>
        <th>ID</th>
        <th>Change</th>
        <th>Final</th>
        <th>Category</th>
        <th>Mode</th>
        <th>Submode</th>
        <th>LowConf</th>
        <th>Drop</th>
        <th>Text</th>
        <th>Image</th>
        <th>Old Overlay</th>
        <th>Refined Overlay</th>
        <th>Old Mask</th>
        <th>Refined Mask</th>
        <th>GT</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <p>{page_nav(page_number, total_pages)}</p>
  <script src="assets/review.js"></script>
</body>
</html>
"""
    (output_dir / f"page_{page_number:04d}.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    results_path = (repo_root / args.results_jsonl).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(results_path)
    reverse = not args.ascending
    if args.sort_by == "sample_id":
        records.sort(key=lambda record: record["sample_id"], reverse=reverse)
    else:
        records.sort(key=lambda record: maybe_float(record.get(args.sort_by)), reverse=reverse)

    total_samples = len(records)
    total_pages = max(1, math.ceil(total_samples / args.page_size))
    summary_path = results_path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    write_assets(output_dir, args.selection_key)
    write_selection_template(output_dir, records)
    write_index(output_dir, total_samples, total_pages, args.page_size, summary, args.title)
    for page_number in range(1, total_pages + 1):
        start = (page_number - 1) * args.page_size
        end = start + args.page_size
        write_page(
            output_dir,
            page_number,
            total_pages,
            records[start:end],
            args.page_size,
            repo_root,
            args.title,
        )

    print(
        json.dumps(
            {
                "total_samples": total_samples,
                "total_pages": total_pages,
                "page_size": args.page_size,
                "index_html": str(output_dir / "index.html"),
                "selection_template": str(output_dir / "selection_template.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
