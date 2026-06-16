import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sample_id_from_path(path: Path) -> str:
    return path.stem


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def parse_cod_domain(sample_id: str) -> str:
    parts = sample_id.split('-')
    if len(parts) >= 4:
        return parts[3]
    return 'Other'


def size_bucket(area_ratio: float, q1: float, q2: float) -> str:
    if area_ratio <= q1:
        return 'small'
    if area_ratio <= q2:
        return 'medium'
    return 'large'


def proportional_allocate(group_sizes, total_needed):
    total_available = sum(group_sizes.values())
    if total_available < total_needed:
        raise ValueError(f'Not enough samples: need {total_needed}, only {total_available}')

    allocation = {}
    fractions = []
    assigned = 0
    for key, size in group_sizes.items():
        raw = total_needed * (size / total_available)
        base = min(size, int(raw))
        allocation[key] = base
        assigned += base
        fractions.append((raw - base, key))

    remaining = total_needed - assigned
    for _, key in sorted(fractions, reverse=True):
        if remaining <= 0:
            break
        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            remaining -= 1

    if remaining > 0:
        for key, size in sorted(group_sizes.items(), key=lambda item: item[1], reverse=True):
            if remaining <= 0:
                break
            spare = size - allocation[key]
            if spare <= 0:
                continue
            take = min(spare, remaining)
            allocation[key] += take
            remaining -= take

    return allocation


def compute_area_ratio(gt_path: Path) -> float:
    gt = np.array(Image.open(gt_path).convert('L'))
    return float((gt > 127).sum()) / float(gt.size)


def build_records(train_image_root: Path, train_gt_root: Path, refined_gt_root: Path):
    records = []
    for img_path in sorted(train_image_root.iterdir()):
        if not img_path.is_file():
            continue
        sample_id = sample_id_from_path(img_path)
        gt_path = train_gt_root / f'{sample_id}.png'
        refined_path = refined_gt_root / f'{sample_id}.png'
        if not gt_path.exists() or not refined_path.exists():
            raise FileNotFoundError(f'Missing GT/refined mask for {sample_id}')
        source = 'CAMO' if sample_id.startswith('camourflage_') else 'COD10K'
        domain = parse_cod_domain(sample_id) if source == 'COD10K' else 'CAMO'
        area_ratio = compute_area_ratio(gt_path)
        records.append({
            'sample_id': sample_id,
            'image_path': str(img_path),
            'gt_path': str(gt_path),
            'refined_gt_path': str(refined_path),
            'source': source,
            'domain': domain,
            'area_ratio': area_ratio,
        })
    return records


def stratified_dev_split(records, seed: int, camo_n: int, cod_n: int):
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for item in records:
        by_source[item['source']].append(item)

    dev_ids = set()

    camo_records = by_source['CAMO']
    camo_areas = sorted(item['area_ratio'] for item in camo_records)
    camo_q1 = camo_areas[len(camo_areas) // 3]
    camo_q2 = camo_areas[(len(camo_areas) * 2) // 3]
    camo_groups = defaultdict(list)
    for item in camo_records:
        camo_groups[size_bucket(item['area_ratio'], camo_q1, camo_q2)].append(item)
    camo_alloc = proportional_allocate({k: len(v) for k, v in camo_groups.items()}, camo_n)
    for bucket, amount in camo_alloc.items():
        choices = camo_groups[bucket][:]
        rng.shuffle(choices)
        dev_ids.update(item['sample_id'] for item in choices[:amount])

    cod_records = by_source['COD10K']
    cod_areas = sorted(item['area_ratio'] for item in cod_records)
    cod_q1 = cod_areas[len(cod_areas) // 3]
    cod_q2 = cod_areas[(len(cod_areas) * 2) // 3]
    cod_groups = defaultdict(list)
    for item in cod_records:
        bucket = size_bucket(item['area_ratio'], cod_q1, cod_q2)
        cod_groups[(item['domain'], bucket)].append(item)
    cod_alloc = proportional_allocate({k: len(v) for k, v in cod_groups.items()}, cod_n)
    for key, amount in cod_alloc.items():
        choices = cod_groups[key][:]
        rng.shuffle(choices)
        dev_ids.update(item['sample_id'] for item in choices[:amount])

    return dev_ids


def read_selected_ids(csv_path: Path):
    selected = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            keep_value = str(row.get('keep', '')).strip().lower()
            if keep_value in {'1', 'true', 'yes', 'y'}:
                selected.append(row['sample_id'].strip())
    return selected


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_csv', required=True)
    parser.add_argument('--train_image_root', required=True)
    parser.add_argument('--train_gt_root', required=True)
    parser.add_argument('--refined_gt_root', required=True)
    parser.add_argument('--dev_root', required=True)
    parser.add_argument('--train_selected_root', required=True)
    parser.add_argument('--seed', type=int, default=20260409)
    parser.add_argument('--camo_dev', type=int, default=100)
    parser.add_argument('--cod_dev', type=int, default=300)
    args = parser.parse_args()

    train_image_root = Path(args.train_image_root)
    train_gt_root = Path(args.train_gt_root)
    refined_gt_root = Path(args.refined_gt_root)
    dev_root = Path(args.dev_root)
    train_selected_root = Path(args.train_selected_root)
    selected_csv = Path(args.selected_csv)

    records = build_records(train_image_root, train_gt_root, refined_gt_root)
    record_by_id = {row['sample_id']: row for row in records}
    dev_ids = stratified_dev_split(records, args.seed, args.camo_dev, args.cod_dev)

    selected_ids = read_selected_ids(selected_csv)
    missing_selected = [sample_id for sample_id in selected_ids if sample_id not in record_by_id]
    if missing_selected:
        raise ValueError(f'Selected CSV contains unknown sample IDs: {missing_selected[:10]}')

    overlap_ids = sorted(set(selected_ids) & dev_ids)
    train_ids = [sample_id for sample_id in selected_ids if sample_id not in dev_ids]

    ensure_clean_dir(dev_root)
    ensure_clean_dir(train_selected_root)
    (train_selected_root / 'Image').mkdir(parents=True, exist_ok=True)
    (train_selected_root / 'GT').mkdir(parents=True, exist_ok=True)

    dev_manifest = []
    for sample_id in sorted(dev_ids):
        row = record_by_id[sample_id]
        dataset_dir = dev_root / row['source']
        image_dst = dataset_dir / 'Image' / Path(row['image_path']).name
        gt_dst = dataset_dir / 'GT' / Path(row['gt_path']).name
        symlink_file(Path(row['image_path']), image_dst)
        symlink_file(Path(row['gt_path']), gt_dst)
        dev_manifest.append({
            'sample_id': sample_id,
            'source': row['source'],
            'domain': row['domain'],
            'area_ratio': row['area_ratio'],
            'image_path': str(image_dst),
            'gt_path': str(gt_dst),
        })

    train_manifest = []
    for sample_id in train_ids:
        row = record_by_id[sample_id]
        image_dst = train_selected_root / 'Image' / Path(row['image_path']).name
        gt_dst = train_selected_root / 'GT' / Path(row['refined_gt_path']).name
        symlink_file(Path(row['image_path']), image_dst)
        symlink_file(Path(row['refined_gt_path']), gt_dst)
        train_manifest.append({
            'sample_id': sample_id,
            'source': row['source'],
            'domain': row['domain'],
            'area_ratio': row['area_ratio'],
            'image_path': str(image_dst),
            'gt_path': str(gt_dst),
        })

    write_manifest(dev_root / 'dev_manifest.jsonl', dev_manifest)
    write_manifest(train_selected_root / 'train_manifest.jsonl', train_manifest)
    write_csv(
        dev_root / 'dev_manifest.csv',
        dev_manifest,
        ['sample_id', 'source', 'domain', 'area_ratio', 'image_path', 'gt_path'],
    )
    write_csv(
        train_selected_root / 'train_manifest.csv',
        train_manifest,
        ['sample_id', 'source', 'domain', 'area_ratio', 'image_path', 'gt_path'],
    )
    write_csv(
        train_selected_root / 'excluded_dev_overlap.csv',
        [{'sample_id': sample_id} for sample_id in overlap_ids],
        ['sample_id'],
    )

    summary = {
        'seed': args.seed,
        'train_total': len(records),
        'dev_total': len(dev_ids),
        'dev_camo': sum(1 for sample_id in dev_ids if sample_id.startswith('camourflage_')),
        'dev_cod10k': sum(1 for sample_id in dev_ids if sample_id.startswith('COD10K-CAM-')),
        'selected_total_from_csv': len(selected_ids),
        'selected_overlap_with_dev': len(overlap_ids),
        'train_selected_total': len(train_ids),
    }
    with (train_selected_root / 'split_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
