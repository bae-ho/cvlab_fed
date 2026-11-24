"""Standalone (non-federated) prototype: train Fed-WAVAD's Model on ONE cctv client
(e.g. the Fight event-split csv) locally, with and without an added PEL4VAD-style
prompt-contrastive loss (utils/tools.py:pel4vad_contrastive_loss), and compare ROC/AP
on the shared cctv test set (same proxy-metric convention fed/server.py already uses
for personalized/per-client evaluation -- one global test split, no per-client held-out
set). This intentionally does not touch fed/ or train.py: it's a cheap sanity check of
whether the new loss is even worth wiring into the federated loop.

Usage:
    python scripts/experiment_pel4vad_client.py --client-class Fight
    python scripts/experiment_pel4vad_client.py --client-class Fight --epochs 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from fed.server import CCTV_LABEL_MAP, UCF_LABEL_MAP
from utils.dataset import UCFDataset
from utils.model import Model
from utils.tools import CLASM, get_batch_label, get_prompt_text, pel4vad_contrastive_loss

EVENT_CLASSES = {
    'cctv': ['Fight', 'Flood', 'Crowd', 'Invasion', 'Gathering', 'Falldown'],
    'ucf': [c for c in UCF_LABEL_MAP if c != 'Normal'],
}
LABEL_MAPS = {'cctv': CCTV_LABEL_MAP, 'ucf': UCF_LABEL_MAP}


def build_loaders(dataset, client_class, batch_size, visual_length):
    train_list = f'./data/list/{dataset}_event/{dataset}_{client_class}.csv'
    normal_loader = DataLoader(
        UCFDataset(visual_length, train_list, False, True), batch_size=batch_size,
        shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(
        UCFDataset(visual_length, train_list, False, False), batch_size=batch_size,
        shuffle=True, drop_last=True)
    test_loader = DataLoader(
        UCFDataset(visual_length, f'./data/list/{dataset}_test.csv', True), batch_size=1, shuffle=False)
    return normal_loader, anomaly_loader, test_loader


def compute_logits_and_features(model, visual, prompt_text):
    """Mirrors Model.forward()'s logic (utils/model.py:273) but also returns the
    intermediate visual/text features pel4vad_contrastive_loss needs."""
    visual_features = model.encode_video(visual)
    text_features = model.encode_text_prompt(prompt_text, visual_features, use_personalization=False)

    text_b = text_features.unsqueeze(0).expand(visual_features.shape[0], -1, -1)
    v_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
    t_norm = text_b / text_b.norm(dim=-1, keepdim=True)
    t_norm = t_norm.permute(0, 2, 1)
    logits = (v_norm @ t_norm.type(v_norm.dtype)) * model.clipmodel.logit_scale.exp()
    return logits, visual_features, text_features


@torch.no_grad()
def evaluate(model, test_loader, gt, device, prompt_text, visual_length):
    model.eval()
    all_probs = []
    for item in test_loader:
        visual = item[0].squeeze(0).to(device)
        length = int(item[2])
        len_cur = length
        if len_cur < visual_length:
            visual = visual.unsqueeze(0)

        logits, _, _ = compute_logits_and_features(model, visual, prompt_text)
        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        prob = 1 - logits[0:len_cur].softmax(dim=-1)[:, 0]
        all_probs.append(prob.cpu())

    all_probs = torch.cat(all_probs).numpy().tolist()
    roc = roc_auc_score(gt, np.repeat(all_probs, 16))
    ap = average_precision_score(gt, np.repeat(all_probs, 16))
    return roc, ap


def run(args, use_pel4vad: bool):
    torch.manual_seed(args.seed)
    device = args.device

    normal_loader, anomaly_loader, test_loader = build_loaders(
        args.dataset, args.client_class, args.batch_size, args.visual_length)

    label_map = {'Normal': 'normal', args.client_class: args.prompt_word}
    prompt_text = get_prompt_text(label_map)  # ['normal', <class prompt word>], normal always index 0
    normal_idx = 0

    model = Model(args.embed_dim, args.visual_length, args.prompt_prefix, args.prompt_postfix,
                  args.visual_width, args.visual_layers, args.visual_head, args.attn_window,
                  device, personalization_type='none').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_batches = min(len(normal_loader), len(anomaly_loader))
    for epoch in range(args.epochs):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        epoch_clasm, epoch_pel = 0.0, 0.0

        for _ in range(n_batches):
            normal_feat, normal_label, normal_len = next(normal_iter)
            anomaly_feat, anomaly_label, anomaly_len = next(anomaly_iter)

            visual = torch.cat([normal_feat, anomaly_feat], dim=0).to(device)
            lengths = torch.cat([normal_len, anomaly_len], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            is_normal = torch.tensor([t == 'Normal' for t in text_labels], device=device)
            sample_class_idx = torch.ones(len(text_labels), dtype=torch.long, device=device)

            logits, visual_features, text_features = compute_logits_and_features(model, visual, prompt_text)

            gt_vec = get_batch_label(text_labels, prompt_text, label_map, args.dataset).to(device)
            loss = CLASM(logits, gt_vec, lengths, device)
            epoch_clasm += loss.item()

            if use_pel4vad:
                pel_loss = pel4vad_contrastive_loss(
                    visual_features, text_features, logits, normal_idx,
                    sample_class_idx, is_normal, temperature=args.pel4vad_temp)
                loss = loss + args.pel4vad_weight * pel_loss
                epoch_pel += pel_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tag = 'pel4vad' if use_pel4vad else 'baseline'
        print(f"  [{tag}] epoch {epoch + 1}/{args.epochs}  CLASM={epoch_clasm / n_batches:.4f}"
              + (f"  PEL4VAD={epoch_pel / n_batches:.4f}" if use_pel4vad else ""))

    gt = np.load(f'./data/gt_{args.dataset}.npy')
    roc, ap = evaluate(model, test_loader, gt, device, prompt_text, args.visual_length)
    return roc, ap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='cctv', choices=['cctv', 'ucf'])
    p.add_argument('--client-class', default='Fight')
    p.add_argument('--seed', type=int, default=88888888)
    p.add_argument('--prompt-word', default=None,
                    help="text used for the class prompt; defaults to the dataset's LABEL_MAP[client-class]")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--pel4vad-weight', type=float, default=1.0)
    p.add_argument('--pel4vad-temp', type=float, default=0.09)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--visual_length', type=int, default=256)
    p.add_argument('--visual_width', type=int, default=512)
    p.add_argument('--visual_layers', type=int, default=2)
    p.add_argument('--visual_head', type=int, default=1)
    p.add_argument('--attn_window', type=int, default=8)
    p.add_argument('--prompt_prefix', type=int, default=10)
    p.add_argument('--prompt_postfix', type=int, default=10)
    args = p.parse_args()

    if args.client_class not in EVENT_CLASSES[args.dataset]:
        p.error(f"--client-class {args.client_class!r} not in {args.dataset}'s classes: "
                f"{EVENT_CLASSES[args.dataset]}")
    if args.prompt_word is None:
        args.prompt_word = LABEL_MAPS[args.dataset][args.client_class]

    print(f"dataset={args.dataset!r} client={args.client_class!r} prompt_word={args.prompt_word!r} "
          f"device={args.device}")
    print(f"NOTE: evaluated on the full shared {args.dataset} test set (all classes + Normal), same "
          "proxy-metric convention as fed/server.py's evaluate_personalized -- this client's "
          "model never saw the other classes during training.\n")

    print("=== baseline (CLASM only) ===")
    base_roc, base_ap = run(args, use_pel4vad=False)
    print(f"baseline: roc={base_roc:.4f} ap={base_ap:.4f}\n")

    print("=== + PEL4VAD contrastive loss ===")
    pel_roc, pel_ap = run(args, use_pel4vad=True)
    print(f"pel4vad : roc={pel_roc:.4f} ap={pel_ap:.4f}\n")

    print(f"delta: roc={pel_roc - base_roc:+.4f}  ap={pel_ap - base_ap:+.4f}")


if __name__ == '__main__':
    main()
