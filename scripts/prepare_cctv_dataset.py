"""Preprocess the AI Hub '지능형 관제 서비스 CCTV 영상 데이터' dataset into the
CLIP-feature + csv-list format Fed-WAVAD/FedPGP already trains on (same layout as
data/VadCLIP-data + data/list/ucf_*).

Source layout (per split in {Training, Validation}):
    <root>/3.개방데이터/1.데이터/<split>/01.원천데이터/{TS,VS}_<category>.zip   (mp4 videos)
    <root>/3.개방데이터/1.데이터/<split>/02.라벨링데이터/{TL,VL}_<category>.zip  (per-video json labels)

Each json label looks like:
    {"metadata": {"frame_count": int, ...},
     "annotations": {"event_class": str, "event_frame": [[start, end], ...], ...}}

Category -> (csv label, is_normal) mapping is inferred from the zip name (see
classify_category): flood level 0 and the pre-Halloween crowd folders are the
"nothing anomalous happening" condition for their scenario despite sharing the
JSON event_class of the abnormal folders, so they are relabeled Normal here.

Pipeline (see --stage):
  extract     (default) walk every zip pair, decode each video, run it through the
              project's own CLIP ViT-B/16 (utils/clip), save one (T, 512) float16
              .npy per video, and append one line to data/list/cctv_manifest.jsonl.
              Resumable: a video already present in the manifest is skipped.
  lists-only  Skip all video/CLIP work and just (re)build cctv_train.csv,
              cctv_test.csv, data/list/cctv_event/cctv_<Class>.csv and gt_cctv.npy
              from an existing data/list/cctv_manifest.jsonl.

Usage:
    pip install opencv-python   # not in requirements.txt's original pin set
    python scripts/prepare_cctv_dataset.py --dry-run       # sanity-check the zip mapping only
    python scripts/prepare_cctv_dataset.py                 # full run (needs a GPU + time)
    python scripts/prepare_cctv_dataset.py --stage lists-only
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `utils.clip`

CLIP_STRIDE = 16  # 1 feature per 16 raw frames, matching UCFClipFeatures / gt_ucf.npy's np.repeat(pred, 16)

SPLIT_DIRS = {
    'Training': ('TS', 'TL'),
    'Validation': ('VS', 'VL'),
}

EVENT_CLASSES = ['Fight', 'Flood', 'Crowd', 'Invasion', 'Gathering', 'Falldown']


def classify_category(name: str):
    """zip name (with the TS_/VS_/TL_/VL_ prefix already stripped) -> (label, is_normal)."""
    if name.startswith('침수'):
        return ('Normal', True) if '0단계' in name else ('Flood', False)
    if name.startswith('인파밀집'):
        if '1주전' in name:
            return 'Normal', True
        if '포함' in name:
            return 'Crowd', False
        raise ValueError(f"unrecognized 인파밀집 category: {name!r}")
    if name.startswith('싸움'):
        return 'Fight', False
    if name.startswith('쓰러짐'):
        return 'Falldown', False
    if name.startswith('군집'):
        return 'Gathering', False
    if name.startswith('침입'):
        return 'Invasion', False
    raise ValueError(f"unrecognized category: {name!r}")


def find_zip_pairs(root: str, split: str):
    """Yields (category_name, label, is_normal, source_zip_path, label_zip_path)."""
    sprefix, lprefix = SPLIT_DIRS[split]
    base = os.path.join(root, '3.개방데이터', '1.데이터', split)
    src_dir = os.path.join(base, '01.원천데이터')
    label_dir = os.path.join(base, '02.라벨링데이터')

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith('.zip'):
            continue
        prefix, name = fname[:-4].split('_', 1)
        assert prefix == lprefix, f"unexpected label zip prefix {fname!r} in {label_dir}"
        label, is_normal = classify_category(name)

        src_name = f"{sprefix}_{name}.zip"
        src_path = os.path.join(src_dir, src_name)
        if not os.path.exists(src_path):
            print(f"  [skip] no matching source zip for {fname}: expected {src_name}")
            continue

        yield name, label, is_normal, src_path, os.path.join(label_dir, fname)


def slugify(name: str) -> str:
    return (name.replace(' ', '-').replace('_', '-')
            .encode('ascii', 'ignore').decode('ascii')) or 'x'


def load_manifest(manifest_path: str):
    entries = []
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def decode_sampled_frames(video_path: str, declared_frame_count: int):
    """Sequentially decode the video and keep the center frame of every non-overlapping
    16-frame window (0-indexed). Sequential read is used instead of seeking to arbitrary
    frame indices because cv2/ffmpeg seeks are unreliable on many consumer-encoded mp4s."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % CLIP_STRIDE == CLIP_STRIDE // 2:
            frames.append(frame[:, :, ::-1])  # BGR -> RGB
        idx += 1
    cap.release()

    actual_frame_count = idx
    if declared_frame_count and abs(actual_frame_count - declared_frame_count) > CLIP_STRIDE:
        print(f"    [warn] decoded {actual_frame_count} frames, json declared {declared_frame_count}")

    num_clips = actual_frame_count // CLIP_STRIDE
    return frames[:num_clips], num_clips, actual_frame_count


def encode_clip_features(frames, clip_model, preprocess, device, batch_size):
    from PIL import Image

    if not frames:
        return np.zeros((0, 512), dtype=np.float16)

    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = [preprocess(Image.fromarray(f)) for f in frames[i:i + batch_size]]
            batch = torch.stack(batch).to(device)
            out = clip_model.encode_image(batch)
            feats.append(out.detach().to(torch.float16).cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_stage(args):
    from utils.clip import clip

    manifest_path = os.path.join(args.out_list_dir, 'cctv_manifest.jsonl')
    existing = load_manifest(manifest_path)
    done_paths = {e['path'] for e in existing}
    print(f"resuming: {len(done_paths)} videos already in manifest")

    clip_model = preprocess = None
    if not args.dry_run:
        device = args.device
        clip_model, preprocess = clip.load(args.clip_name, device=device)
        clip_model.eval()

    scratch_dir = args.scratch_dir or tempfile.mkdtemp(prefix='cctv_prep_')
    os.makedirs(scratch_dir, exist_ok=True)
    manifest_f = open(manifest_path, 'a', encoding='utf-8')

    try:
        for split in ('Training', 'Validation'):
            for category, label, is_normal, src_zip_path, label_zip_path in find_zip_pairs(args.root, split):
                print(f"[{split}] {category} -> label={label} is_normal={is_normal}")
                out_dir = os.path.join(args.out_features_dir, label)
                os.makedirs(out_dir, exist_ok=True)

                with zipfile.ZipFile(label_zip_path) as lz, zipfile.ZipFile(src_zip_path) as sz:
                    src_names = {os.path.basename(n): n for n in sz.namelist() if n.lower().endswith('.mp4')}
                    json_names = sorted(n for n in lz.namelist() if n.lower().endswith('.json'))
                    if args.limit:
                        json_names = json_names[:args.limit]

                    for jn in json_names:
                        stem = os.path.splitext(os.path.basename(jn))[0]
                        mp4_name = f"{stem}.mp4"
                        if mp4_name not in src_names:
                            print(f"  [skip] no video for {jn}")
                            continue

                        out_name = f"{slugify(category)}_{stem}.npy"
                        out_path = os.path.join(out_dir, out_name)
                        rel_path = os.path.relpath(out_path, args.repo_root).replace(os.sep, '/')

                        if rel_path in done_paths:
                            continue

                        meta = json.loads(lz.read(jn).decode('utf-8'))
                        frame_count = meta['metadata']['frame_count']
                        event_frame = meta['annotations'].get('event_frame', [])

                        if os.path.exists(out_path):
                            num_clips = int(np.load(out_path, mmap_mode='r').shape[0])
                        elif args.dry_run:
                            num_clips = max(frame_count // CLIP_STRIDE, 0)
                        else:
                            video_bytes = sz.read(src_names[mp4_name])
                            tmp_path = os.path.join(scratch_dir, mp4_name)
                            with open(tmp_path, 'wb') as tf:
                                tf.write(video_bytes)
                            try:
                                frames, num_clips, _ = decode_sampled_frames(tmp_path, frame_count)
                                feats = encode_clip_features(
                                    frames, clip_model, preprocess, args.device, args.clip_batch_size)
                                np.save(out_path, feats)
                            finally:
                                os.remove(tmp_path)

                        entry = {
                            'path': rel_path,
                            'label': 'Normal' if is_normal else label,
                            'split': split,
                            'is_normal': is_normal,
                            'num_clips': num_clips,
                            'event_frame': event_frame,
                        }
                        manifest_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        manifest_f.flush()
                        done_paths.add(rel_path)
    finally:
        manifest_f.close()
        if not args.scratch_dir:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return manifest_path


def build_lists(args, manifest_path):
    entries = load_manifest(manifest_path)
    print(f"building lists from {len(entries)} manifest entries")

    os.makedirs(args.out_list_dir, exist_ok=True)
    event_dir = os.path.join(args.out_list_dir, 'cctv_event')
    os.makedirs(event_dir, exist_ok=True)

    train_rows = [e for e in entries if e['split'] == 'Training']
    test_rows = [e for e in entries if e['split'] == 'Validation']

    def write_csv(path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('path,label\n')
            for r in rows:
                f.write(f"{r['path']},{r['label']}\n")
        print(f"  wrote {path} ({len(rows)} rows)")

    write_csv(os.path.join(args.out_list_dir, 'cctv_train.csv'), train_rows)
    write_csv(os.path.join(args.out_list_dir, 'cctv_test.csv'), test_rows)

    for cls in EVENT_CLASSES:
        rows = [r for r in train_rows if r['label'] == cls or r['is_normal']]
        write_csv(os.path.join(event_dir, f'cctv_{cls}.csv'), rows)

    # gt_cctv.npy: frame-level GT for the Validation split, concatenated in the exact
    # row order written to cctv_test.csv above (test_loader iterates the csv in file order).
    segments = []
    for r in test_rows:
        length = r['num_clips'] * CLIP_STRIDE
        gt = np.zeros(length, dtype=np.float32)
        if not r['is_normal']:
            for start, end in r['event_frame']:
                s = max(0, int(start) - 1)
                e = min(length - 1, int(end) - 1)
                if e >= s:
                    gt[s:e + 1] = 1.0
        segments.append(gt)

    gt_all = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    gt_path = os.path.join(args.data_dir, 'gt_cctv.npy')
    np.save(gt_path, gt_all)
    print(f"  wrote {gt_path} ({gt_all.shape[0]} frames, {int(gt_all.sum())} anomalous)")

    label_counts = defaultdict(int)
    for e in entries:
        label_counts[(e['split'], e['label'])] += 1
    print("label counts:", dict(label_counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='./07.지능형 관제 서비스 CCTV 영상 데이터',
                         help="AI Hub dataset root (the folder containing '3.개방데이터')")
    parser.add_argument('--repo-root', default='.', help="Fed-WAVAD repo root")
    parser.add_argument('--stage', choices=['extract', 'lists-only'], default='extract')
    parser.add_argument('--dry-run', action='store_true',
                         help="validate zip pairing/category mapping and manifest counts only; "
                              "no CLIP model load, no video decode")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clip-name', default='ViT-B/16')
    parser.add_argument('--clip-batch-size', type=int, default=64)
    parser.add_argument('--scratch-dir', default=None,
                         help="dir for temporary extracted mp4s; default is an auto-cleaned tempdir")
    parser.add_argument('--limit', type=int, default=None,
                         help="cap videos processed per category zip (smoke testing)")
    args = parser.parse_args()

    args.repo_root = os.path.abspath(args.repo_root)
    args.data_dir = os.path.join(args.repo_root, 'data')
    args.out_features_dir = os.path.join(args.data_dir, 'CCTVClipFeatures')
    args.out_list_dir = os.path.join(args.data_dir, 'list')
    os.makedirs(args.out_features_dir, exist_ok=True)
    os.makedirs(args.out_list_dir, exist_ok=True)

    if args.stage == 'extract':
        manifest_path = extract_stage(args)
    else:
        manifest_path = os.path.join(args.out_list_dir, 'cctv_manifest.jsonl')

    build_lists(args, manifest_path)


if __name__ == '__main__':
    main()
