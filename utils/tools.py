import torch
import numpy as np
import torch.nn.functional as F


def get_batch_label(texts, prompt_text, label_map: dict, dataset):
    label_vectors = torch.zeros(0)
    if dataset in ('ucf', 'cctv'):
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    # xd
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors


def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text


def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1

    return mask.bool()


def random_extract(feat, t_max):
    r = np.random.randint(feat.shape[0] - t_max)
    return feat[r: r + t_max, :]


def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg is True:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate(
                    [split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate(
                    [split_feat,
                     pad(feat[i * length:i * length + length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length


def fedpgp_contrastive_loss(text_features_global, text_features_personalized, class_features, temperature):
    """Pulls the un-personalized dynamic prompt (z_global) toward CLIP's own
    handcrafted-class embedding (z_handcrafted), while the personalized prompt
    (z_personalized) is free to diverge from it toward client-specific structure.

    Positive pair: (z_global, z_handcrafted). Negative pair: (z_personalized, z_handcrafted).
    Both are already per-class vectors (no batch dimension), so reduction is a mean over
    classes. z_handcrafted is computed under torch.no_grad() upstream, so gradients only
    flow through z_global (into the shared prompt generator) and z_personalized (into the
    shared prompt generator + the personalized adapter).

    -log[ exp(sim_pos/t) / (exp(sim_pos/t) + exp(sim_neg/t)) ] == softplus(sim_neg/t - sim_pos/t)
    """
    z_global = F.normalize(text_features_global, dim=-1)
    z_personalized = F.normalize(text_features_personalized, dim=-1)
    z_handcrafted = class_features

    sim_pos = (z_global * z_handcrafted).sum(dim=-1) / temperature
    sim_neg = (z_personalized * z_handcrafted).sum(dim=-1) / temperature

    return F.softplus(sim_neg - sim_pos).mean()


def pel4vad_contrastive_loss(visual_features, text_features, logits, normal_idx,
                              sample_class_idx, is_normal, temperature=0.09, scale=10.0):
    """PEL4VAD-style video-to-prompt contrastive loss (Pu et al., "Learning Prompt-Enhanced
    Context Features for Weakly-Supervised Video Anomaly Detection", arXiv:2306.14451),
    ported from the official get_cas/create_logits/KLV_loss in
    https://github.com/yujiangpu20/PEL4VAD.

    Uses the model's own per-snippet class similarity (logits) as a soft-attention weight to
    pool visual_features into one "most anomalous" vector and one "most normal" vector per
    video, then contrasts those pooled vectors against the class prompt embeddings the video
    should/shouldn't align with. A Normal-labeled video only contributes the anomalous-pooled
    vector paired with the normal prompt (nothing else to contrast it against); an
    abnormal-labeled video contributes that pooled vector paired with its own class prompt
    *and* the normal-pooled vector paired with the normal prompt, so the two poolings of the
    same weakly-labeled video get pushed toward different prompts.

    visual_features: (B, T, C) -- Model.encode_video output
    text_features: (num_classes, C) -- Model.encode_text_prompt output for this batch
    logits: (B, T, num_classes) -- Model.forward's per-snippet class-similarity logits
    normal_idx: int, index of the 'normal' class in text_features/logits' class dim
    sample_class_idx: LongTensor (B,), index into text_features of each sample's own class
        (only read for samples where is_normal is False)
    is_normal: BoolTensor (B,)
    """
    score = 1.0 - F.softmax(logits.float(), dim=-1)[..., normal_idx]  # (B, T), higher = more anomalous
    score = score.clamp(0.0, 1.0)

    abn_w = torch.exp(scale * score) - 1.0
    abn_w = abn_w / abn_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    nor_w = torch.exp(scale * (1.0 - score)) - 1.0
    nor_w = nor_w / nor_w.sum(dim=1, keepdim=True).clamp_min(1e-6)

    visual_features = visual_features.float()
    abn_feat = torch.einsum('bt,btc->bc', abn_w, visual_features)
    nor_feat = torch.einsum('bt,btc->bc', nor_w, visual_features)

    text_features = text_features.float()
    video_feats, token_feats, video_labels = [], [], []
    for b in range(visual_features.shape[0]):
        if bool(is_normal[b]):
            video_feats.append(abn_feat[b])
            token_feats.append(text_features[normal_idx])
            video_labels.append(0)
        else:
            cls = int(sample_class_idx[b])
            video_feats.append(abn_feat[b])
            token_feats.append(text_features[cls])
            video_labels.append(cls + 1)

            video_feats.append(nor_feat[b])
            token_feats.append(text_features[normal_idx])
            video_labels.append(0)

    video_feats = torch.stack(video_feats)
    token_feats = torch.stack(token_feats)
    video_labels = torch.tensor(video_labels, device=visual_features.device)

    v = F.normalize(video_feats, dim=-1)
    t = F.normalize(token_feats, dim=-1)
    sim = (v @ t.t()) / temperature

    same_label = (video_labels.unsqueeze(0) == video_labels.unsqueeze(1)).float()  # multi-positive target
    log_prob = F.log_softmax(sim, dim=1)
    target = F.softmax(same_label * 10.0, dim=1)

    return F.kl_div(log_prob, target, reduction='batchmean')


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


