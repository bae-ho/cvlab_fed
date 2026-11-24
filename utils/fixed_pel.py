"""Fixed full-class text anchors for the FedAvg PEL experiment.

This is an adaptation of PEL4VAD, not its original batch-pair loss.
"""
import hashlib
import json
import math

import torch
import torch.nn.functional as F


def build_text_bank(model, label_map, prompts_path=None, tokenize=None):
    prompts = {key: [value] for key, value in label_map.items()}
    if prompts_path:
        with open(prompts_path, encoding="utf-8") as stream:
            prompts = json.load(stream)
        if set(prompts) != set(label_map):
            raise ValueError("PEL prompt JSON must contain exactly the dataset class keys")
    for values in prompts.values():
        if not isinstance(values, list) or not values or any(
                not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Each class needs a nonempty list of text prompts")
    if tokenize is None:
        from utils.clip import clip
        tokenize = clip.tokenize
    encoder = model.clipmodel
    device = next(encoder.parameters()).device
    modes = [(module, module.training) for module in encoder.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        encoder.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            features = []
            for key in label_map:
                encoded = encoder.encode_text_original(tokenize(prompts[key]).to(device)).float()
                features.append(F.normalize(F.normalize(encoded, dim=-1).mean(0), dim=0))
            bank = torch.stack(features).detach()
    finally:
        for module, mode in modes:
            module.training = mode
    if not torch.isfinite(bank).all() or (bank.norm(dim=-1) == 0).any():
        raise ValueError("Invalid fixed text embeddings")
    metadata = {"classes": list(label_map), "prompts": prompts,
                "encoder": "frozen CLIP ViT-B/16 encode_text_original",
                "pooling": "mean of normalized prompt embeddings, then normalize",
                "sha256": hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest()}
    return bank, metadata


def single_class_indices(labels):
    if labels.ndim != 2 or not torch.isfinite(labels).all() or not (
            ((labels == 0) | (labels == 1)).all() and (labels.sum(-1) == 1).all()):
        raise ValueError("PEL requires exactly one known class per video")
    return labels.argmax(-1)


def fixed_bank_pel_loss(features, logits, lengths, bank, class_indices,
                        normal_idx=0, temperature=0.09, scale=10.0, diagnostics=None,
                        pooling='score', class_balance='none'):
    if pooling not in ('score', 'uniform') or class_balance not in ('none', 'batch'):
        raise ValueError('Invalid PEL pooling or class balancing')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("PEL temperature and scale must be finite and positive")
    if features.ndim != 3 or logits.ndim != 3 or bank.ndim != 2:
        raise ValueError("Expected BxTxD features, BxTxC logits and CxD bank")
    b, t, d = features.shape
    c = bank.shape[0]
    if b == 0 or logits.shape != (b, t, c) or bank.shape[1] != d or not 0 <= normal_idx < c:
        raise ValueError("Incompatible PEL shapes or normal class")
    if lengths.shape != (b,) or lengths.is_floating_point() or (lengths <= 0).any() or (lengths > t).any():
        raise ValueError("PEL lengths must be integer values in [1, T]")
    if class_indices.shape != (b,) or class_indices.is_floating_point() or (class_indices < 0).any() or (class_indices >= c).any():
        raise ValueError("Invalid PEL class indices")
    if any(not torch.isfinite(x).all() for x in (features, logits, bank)):
        raise ValueError("PEL inputs must be finite")
    mask = torch.arange(t, device=features.device)[None, :] < lengths[:, None]
    scores = 1 - logits.detach().float().softmax(-1)[..., normal_idx]

    def pool(score):
        # Subtracting the row maximum keeps exponential weights finite.
        x = score * scale
        maximum = x.masked_fill(~mask, -torch.inf).max(-1, keepdim=True).values
        weights = (torch.exp(x - maximum) - torch.exp(-maximum)) * mask
        total = weights.sum(-1, keepdim=True)
        uniform = mask.float() / lengths[:, None]
        weights = torch.where(total > 1e-8, weights / total.clamp_min(1e-8), uniform)
        if pooling == 'uniform':
            weights = uniform
        if diagnostics is not None:
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
            diagnostics.setdefault('pooling', []).append({
                'entropy_mean': entropy.detach().mean().item(),
                'max_weight_mean': weights.detach().max(-1).values.mean().item()})
        return torch.einsum("bt,btd->bd", weights, features.float())

    foreground, background = pool(scores), pool(1 - scores)
    abnormal = class_indices != normal_idx
    visual = torch.cat([foreground, background[abnormal]])
    targets = torch.cat([class_indices, torch.full_like(class_indices[abnormal], normal_idx)])
    similarity = F.normalize(visual, dim=-1) @ F.normalize(bank.detach().float(), dim=-1).T / temperature
    distribution = (F.one_hot(targets.long(), c).float() * 10).softmax(-1)
    loss = F.kl_div(similarity.log_softmax(-1), distribution, reduction="batchmean")
    if class_balance == 'batch':
        per_row = F.kl_div(similarity.log_softmax(-1), distribution, reduction='none').sum(-1)
        frequencies = torch.bincount(targets, minlength=c)
        row_weights = frequencies[targets].float().reciprocal()
        row_weights = row_weights / row_weights.mean()
        loss = (per_row * row_weights).mean()
    if diagnostics is not None:
        with torch.no_grad():
            per_row = F.kl_div(similarity.log_softmax(-1), distribution, reduction='none').sum(-1)
            effective_weights = row_weights if class_balance == 'batch' else torch.ones_like(per_row)
            diagnostics['class_balance'] = class_balance
            diagnostics['pooling_mode'] = pooling
            diagnostics['class_terms'] = {
                str(int(cls)): {'count': int((targets == cls).sum()),
                               'mean_kl': per_row[targets == cls].mean().item(),
                               'weighted_contribution': (per_row[targets == cls] *
                                                         effective_weights[targets == cls]).sum().item() / len(targets)}
                for cls in targets.unique()}
            cosine = similarity * temperature
            diagnostics['foreground_target_cosine_mean'] = cosine[:b].gather(1, class_indices[:, None]).mean().item()
            diagnostics['background_normal_cosine_mean'] = (
                cosine[b:, normal_idx].mean().item() if abnormal.any() else None)
    return loss
