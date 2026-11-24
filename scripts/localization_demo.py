"""시간축 이상 구간 Localization 데모.

학습된 체크포인트로 테스트 영상마다 프레임 단위 이상 점수를 계산하고,
"언제(구간)"와 "무엇(이상 클래스)"을 함께 보여주는 결과물을 만든다.
inference.py와 같은 모델·전처리·점수 정의(1 - P(normal), 스니펫당 16프레임)를 사용하므로
전체 ROC-AUC/AP 값은 inference.py 결과와 일치해야 한다.

출력 (--out_dir 아래):
    plots/<영상>.png        영상별 이상 점수 곡선 + GT 이상 구간(빨강) + 예측 구간(초록)
    segments.csv            예측 이상 구간 목록 (시작/끝 프레임·초, 예측 클래스, 최대 점수)
    per_video.csv           영상별 정답 라벨, 예측 클래스, 프레임 IoU, 영상 단위 AUC
    summary.json            전체 ROC-AUC/AP, threshold, 평균 IoU 등
    videos/<영상>_demo.mp4   (--video_dir를 준 경우만) 원본 영상에 점수·판정 오버레이

사용 예 (저장소 루트에서):
    # UCF-Crime, 이상 영상 위주로 그래프 저장
    python scripts/localization_demo.py --dataset ucf --checkpoint model.pth --only_anomaly

    # 특정 영상만, 원본 영상 위에 오버레이까지
    python scripts/localization_demo.py --dataset ucf --checkpoint model.pth \
        --videos Arson011 Fighting033 --video_dir /path/to/UCF_Crimes/Videos

    # 개인화 모델로 테스트할 때는 inference.py와 같은 인자를 그대로 사용
    python scripts/localization_demo.py --checkpoint model.pth --inference_mode personalized \
        --personalization_type low_rank --personalized_checkpoint_dir save/run --personalized_client_id 0
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

FRAMES_PER_SNIPPET = 16  # inference.py의 np.repeat(ap2, 16)과 동일


# ----------------------------------------------------------------------------
# 후처리 (torch 없이 동작하는 순수 함수: tests/test_localization_demo.py에서 검증)
# ----------------------------------------------------------------------------
def smooth(scores, window):
    """이동평균 smoothing. window<=1이면 그대로 반환."""
    scores = np.asarray(scores, dtype=np.float64)
    if window <= 1 or len(scores) == 0:
        return scores
    kernel = np.ones(window) / window
    padded = np.pad(scores, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def binary_to_segments(mask):
    """0/1 배열 -> [(start, end_exclusive), ...]"""
    mask = np.asarray(mask).astype(np.int8)
    if mask.size == 0:
        return []
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def scores_to_segments(scores, threshold, min_len=1, merge_gap=0):
    """스니펫 점수 -> 예측 이상 구간(스니펫 단위). 짧은 틈은 병합, 짧은 구간은 제거."""
    segs = binary_to_segments(np.asarray(scores) >= threshold)
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def segments_to_mask(segments, length):
    mask = np.zeros(length, dtype=bool)
    for s, e in segments:
        mask[s:e] = True
    return mask


def frame_iou(pred_mask, gt_mask):
    """프레임 단위 IoU. GT와 예측이 모두 비어 있으면 None(정상 영상)."""
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return None
    return float(np.logical_and(pred_mask, gt_mask).sum() / union)


def video_name_from_path(path):
    """'.../Arson011_x264__0.npy' -> 'Arson011_x264'"""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return re.sub(r'__\d+$', '', stem)


# ----------------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------------
def plot_video(out_path, name, frame_scores, gt_frames, pred_segments_f, threshold, title_extra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(frame_scores)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for s, e in binary_to_segments(gt_frames):
        ax.axvspan(s, e, color='#e74c3c', alpha=0.22, lw=0)
    ax.plot(x, frame_scores, color='#1f4e9c', lw=1.6, label='anomaly score')
    ax.axhline(threshold, color='gray', ls='--', lw=1, label=f'threshold {threshold:.2f}')
    for s, e in pred_segments_f:
        ax.hlines(-0.06, s, e, colors='#27ae60', lw=6)
    # 범례용 더미
    ax.fill_between([], [], color='#e74c3c', alpha=0.22, label='GT anomaly')
    ax.plot([], [], color='#27ae60', lw=6, label='predicted segment')
    ax.set_xlim(0, max(n - 1, 1))
    ax.set_ylim(-0.12, 1.05)
    ax.set_xlabel('frame')
    ax.set_ylabel('score')
    ax.set_title(f'{name}   {title_extra}', fontsize=10, loc='left', pad=18)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=4, frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def find_video_file(video_dir, name):
    exts = ('.mp4', '.avi', '.mkv', '.mov')
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(exts) and os.path.splitext(f)[0] == name:
                return os.path.join(root, f)
    return None


def render_overlay(video_path, out_path, frame_scores, gt_frames, frame_classes, threshold):
    """원본 영상 하단에 점수 그래프 strip, 상단에 판정 배너를 그린다."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  [skip] 영상을 열 수 없음: {video_path}')
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    strip_h = max(60, h // 5)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h + strip_h))
    n = len(frame_scores)

    # 정적인 strip 배경(GT 구간 + 점수 곡선)을 미리 그려둔다
    strip = np.full((strip_h, w, 3), 255, np.uint8)
    xs = (np.arange(n) * (w - 1) / max(n - 1, 1)).astype(int)
    for s, e in binary_to_segments(gt_frames):
        cv2.rectangle(strip, (xs[s], 0), (xs[min(e, n) - 1], strip_h - 1), (200, 200, 255), -1)
    ys = (strip_h - 4 - np.clip(frame_scores, 0, 1) * (strip_h - 8)).astype(int)
    cv2.polylines(strip, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (156, 78, 31), 2)
    ty = int(strip_h - 4 - threshold * (strip_h - 8))
    cv2.line(strip, (0, ty), (w - 1, ty), (150, 150, 150), 1)

    font = max(0.35, min(0.8, w / 640))  # UCF-Crime(320x240) 같은 작은 영상에서도 글자가 잘리지 않게
    thick = 1 if font < 0.6 else 2
    bar_h = int(30 * font / 0.7) + 4
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k = min(i, n - 1)
        score = float(frame_scores[k])
        anomalous = score >= threshold
        color = (40, 40, 220) if anomalous else (60, 160, 60)
        text = f'ANOMALY: {frame_classes[k]}  {score:.2f}' if anomalous else f'normal  {score:.2f}'
        cv2.rectangle(frame, (0, 0), (w, bar_h), color, -1)
        cv2.putText(frame, text, (6, bar_h - 7), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)
        if anomalous:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)
        s = strip.copy()
        cv2.line(s, (xs[k], 0), (xs[k], strip_h - 1), (0, 0, 0), 1)
        writer.write(np.vstack([frame, s]))
        i += 1
    cap.release()
    writer.release()
    return True


# ----------------------------------------------------------------------------
# 모델 추론 (inference.py와 같은 길이 분할 규칙)
# ----------------------------------------------------------------------------
def split_lengths(length, max_len=256):
    import torch
    lengths = torch.zeros(int(length / max_len) + 1)
    for j in range(int(length / max_len) + 1):
        if j == 0 and length < max_len:
            lengths[j] = length
        elif j == 0 and length > max_len:
            lengths[j] = max_len
            length -= max_len
        elif length > max_len:
            lengths[j] = max_len
            length -= max_len
        else:
            lengths[j] = length
    return lengths.to(int)


def predict_video(model, visual, length, prompt_text, device, use_personalization, visual_length=256):
    """반환: (스니펫별 이상 점수 [T], 스니펫별 클래스 확률 [T, C])"""
    import torch
    visual = visual.squeeze(0).to(device)
    length = int(length)
    if length < visual_length:
        visual = visual.unsqueeze(0)
    lengths = split_lengths(length)
    logits = model(visual, prompt_text, lengths, use_personalization=use_personalization)
    logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])[:length]
    probs = logits.softmax(dim=-1)
    score = 1 - probs[:, 0]
    return score.cpu().numpy(), probs.cpu().numpy()


def load_model(args, device):
    import torch
    from utils.model import Model

    model = Model(args.embed_dim, args.visual_length, args.prompt_prefix,
                  args.prompt_postfix, args.visual_width, args.visual_layers,
                  args.visual_head, args.attn_window, device,
                  personalization_type=args.personalization_type,
                  personalized_rank=args.personalized_rank,
                  personalized_alpha=args.personalized_alpha,
                  pel_projection=args.pel_projection).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    use_personalization = args.inference_mode == 'personalized'
    if use_personalization:
        if args.personalization_type == 'none' or args.personalized_checkpoint_dir is None:
            raise ValueError('--inference_mode personalized 에는 --personalization_type 과 '
                             '--personalized_checkpoint_dir 가 필요합니다 (inference.py와 동일)')
        path = os.path.join(args.personalized_checkpoint_dir,
                            f'client_{args.personalized_client_id}_personalized.pth')
        model.load_personalized_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, use_personalization


def build_parser():
    from utils import config
    p = config.parser
    g = p.add_argument_group('localization demo')
    g.add_argument('--out_dir', default='artifacts/localization_demo')
    g.add_argument('--threshold', type=float, default=0.5, help='이상 판정 점수 기준')
    g.add_argument('--smooth', type=int, default=1, help='스니펫 단위 이동평균 창 크기 (1=끔)')
    g.add_argument('--min_seg', type=int, default=2, help='이보다 짧은 예측 구간(스니펫)은 제거')
    g.add_argument('--merge_gap', type=int, default=1, help='이 이하의 틈(스니펫)은 하나의 구간으로 병합')
    g.add_argument('--videos', nargs='*', default=None, help='이름 일부가 일치하는 영상만 처리 (예: Arson011)')
    g.add_argument('--only_anomaly', action='store_true', help='정답이 Normal인 영상은 건너뜀')
    g.add_argument('--max_videos', type=int, default=None)
    g.add_argument('--video_dir', default=None, help='원본 영상 폴더. 주면 오버레이 mp4 생성')
    g.add_argument('--fps', type=float, default=30.0, help='segments.csv의 초 단위 환산용 fps')
    return p


def main():
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    from fed.server import CCTV_LABEL_MAP, UCF_LABEL_MAP, XD_LABEL_MAP
    from utils.dataset import UCFDataset, XDDataset
    from utils.tools import get_prompt_text

    args = build_parser().parse_args()
    if args.checkpoint is None:
        raise SystemExit('--checkpoint 를 지정하세요')

    test_list = {'ucf': './data/list/ucf_test.csv', 'cctv': './data/list/cctv_test.csv',
                 'xd': './data/list/xd_test.csv'}[args.dataset]
    gt_path = {'ucf': './data/gt_ucf.npy', 'cctv': './data/gt_cctv.npy', 'xd': './data/gt_xd.npy'}[args.dataset]
    label_map = {'ucf': UCF_LABEL_MAP, 'xd': XD_LABEL_MAP, 'cctv': CCTV_LABEL_MAP}[args.dataset]
    prompt_text = get_prompt_text(dict(label_map))

    Dataset = XDDataset if args.dataset == 'xd' else UCFDataset
    test_dataset = Dataset(args.visual_length, test_list, True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    df = pd.read_csv(test_list)
    gt_all = np.load(gt_path)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    model, use_personalization = load_model(args, device)

    plot_dir = os.path.join(args.out_dir, 'plots')
    vid_dir = os.path.join(args.out_dir, 'videos')
    os.makedirs(plot_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(vid_dir, exist_ok=True)

    all_scores, seg_rows, video_rows = [], [], []
    offset, shown = 0, 0
    with torch.no_grad():
        for i, (visual, label, length) in enumerate(test_loader):
            score, probs = predict_video(model, visual, length, prompt_text, device, use_personalization,
                                         args.visual_length)
            n_snip = len(score)
            n_frames = n_snip * FRAMES_PER_SNIPPET
            gt_frames = gt_all[offset: offset + n_frames].astype(bool)
            offset += n_frames
            all_scores.append(score)  # 전체 AUC/AP는 smoothing 전 원점수로 계산 (inference.py와 동일)

            label = label[0] if isinstance(label, (list, tuple)) else str(label)
            name = video_name_from_path(df.loc[i, 'path'])
            if args.videos and not any(v in name for v in args.videos):
                continue
            if args.only_anomaly and label in ('Normal', 'A'):
                continue
            if args.max_videos is not None and shown >= args.max_videos:
                continue
            shown += 1

            s = smooth(score, args.smooth)
            segs = scores_to_segments(s, args.threshold, args.min_seg, args.merge_gap)
            frame_scores = np.repeat(s, FRAMES_PER_SNIPPET)
            segs_f = [(a * FRAMES_PER_SNIPPET, b * FRAMES_PER_SNIPPET) for a, b in segs]
            pred_mask = segments_to_mask(segs_f, n_frames)
            iou = frame_iou(pred_mask, gt_frames)

            # 이상 클래스: 클래스 0(normal)을 제외한 확률 중 최댓값
            snip_cls = probs[:, 1:].argmax(axis=1) + 1
            frame_classes = np.repeat([prompt_text[c] for c in snip_cls], FRAMES_PER_SNIPPET)
            if segs:
                in_seg = segments_to_mask(segs, n_snip)
                video_cls = prompt_text[int(probs[in_seg, 1:].mean(0).argmax()) + 1]
            else:
                video_cls = prompt_text[0]

            for a, b in segs:
                seg_rows.append({
                    'video': name, 'gt_label': label, 'pred_class': prompt_text[int(probs[a:b, 1:].mean(0).argmax()) + 1],
                    'start_frame': a * FRAMES_PER_SNIPPET, 'end_frame': b * FRAMES_PER_SNIPPET,
                    'start_sec': round(a * FRAMES_PER_SNIPPET / args.fps, 2),
                    'end_sec': round(b * FRAMES_PER_SNIPPET / args.fps, 2),
                    'max_score': round(float(s[a:b].max()), 4)})

            vid_auc = None
            if 0 < gt_frames.sum() < len(gt_frames):
                vid_auc = float(roc_auc_score(gt_frames, np.repeat(score, FRAMES_PER_SNIPPET)))
            video_rows.append({'video': name, 'gt_label': label, 'pred_class': video_cls,
                               'num_segments': len(segs), 'frame_iou': None if iou is None else round(iou, 4),
                               'video_auc': None if vid_auc is None else round(vid_auc, 4),
                               'max_score': round(float(s.max()), 4)})

            extra = f'GT: {label}   pred: {video_cls}' + ('' if iou is None else f'   IoU {iou:.2f}')
            plot_video(os.path.join(plot_dir, f'{name}.png'), name, frame_scores, gt_frames,
                       segs_f, args.threshold, extra)

            if args.video_dir:
                vpath = find_video_file(args.video_dir, name)
                if vpath is None:
                    print(f'  [skip] 원본 영상 없음: {name}')
                else:
                    render_overlay(vpath, os.path.join(vid_dir, f'{name}_demo.mp4'),
                                   frame_scores, gt_frames, frame_classes, args.threshold)
            print(f'[{shown}] {name}: GT={label} pred={video_cls} segments={len(segs)}'
                  + ('' if iou is None else f' IoU={iou:.3f}'))

    if offset != len(gt_all):
        print(f'[경고] GT 길이({len(gt_all)})와 예측 프레임 수({offset})가 다릅니다. '
              'test csv와 gt 파일이 같은 순서/버전인지 확인하세요.')

    scores_f = np.repeat(np.concatenate(all_scores), FRAMES_PER_SNIPPET)
    m = min(len(scores_f), len(gt_all))
    ious = [r['frame_iou'] for r in video_rows if r['frame_iou'] is not None and r['gt_label'] not in ('Normal', 'A')]
    summary = {
        'checkpoint': args.checkpoint, 'dataset': args.dataset, 'inference_mode': args.inference_mode,
        'roc_auc': float(roc_auc_score(gt_all[:m], scores_f[:m])),
        'ap': float(average_precision_score(gt_all[:m], scores_f[:m])),
        'threshold': args.threshold, 'smooth': args.smooth, 'min_seg': args.min_seg, 'merge_gap': args.merge_gap,
        'videos_rendered': len(video_rows),
        'mean_frame_iou_anomalous': None if not ious else round(float(np.mean(ious)), 4),
    }

    def write_csv(path, rows, fields):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_csv(os.path.join(args.out_dir, 'segments.csv'), seg_rows,
              ['video', 'gt_label', 'pred_class', 'start_frame', 'end_frame', 'start_sec', 'end_sec', 'max_score'])
    write_csv(os.path.join(args.out_dir, 'per_video.csv'), video_rows,
              ['video', 'gt_label', 'pred_class', 'num_segments', 'frame_iou', 'video_auc', 'max_score'])
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nROC-AUC {summary['roc_auc']:.4f}  AP {summary['ap']:.4f}  "
          f"mean IoU(anomalous) {summary['mean_frame_iou_anomalous']}")
    print(f'결과 저장 위치: {args.out_dir}')


if __name__ == '__main__':
    main()
