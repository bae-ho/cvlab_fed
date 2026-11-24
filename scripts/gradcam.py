"""학습된 이상 탐지 모델의 판단 근거를 화면 위에 표시하는 Grad-CAM.

모델 입력은 프레임마다 CLIP ViT-B/16이 만든 512차원 특징이다. 이 모듈은
  1) 설명할 스니펫 t의 원본 프레임을 같은 CLIP 영상 인코더에 직접 통과시켜 특징을 만들고
  2) 그 영상의 나머지 스니펫은 저장된 특징을 그대로 둔 채 t 자리만 새 특징으로 바꿔 모델에 넣은 뒤
  3) 스니펫 t의 이상 점수 (1 - P(normal)) 를 CLIP 마지막 블록 입력의 패치 토큰(14x14)까지 미분해
  4) Grad-CAM (패치 활성 x 채널별 평균 기울기, ReLU) 지도를 만든다.
즉 "이 모델의 이상 점수를 올린 화면 영역"이며, 정답 위치를 학습한 결과는 아니다.

직접 만든 특징이 저장된 특징과 다르면 설명이 부정확해지므로, 영상마다 프레임 위치(오프셋)와
자르기 방식을 저장 특징과의 코사인 유사도로 자동 선택하고 그 값을 함께 기록한다.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
STRIDE = 16

# (이름, 짧은 변 리사이즈 크기 또는 None=정사각형으로 늘림, 자르기 위치)
CROPS = [('clip_center', 224, 'c'), ('full', None, None),
         ('256_center', 256, 'c'), ('256_tl', 256, 'tl'), ('256_tr', 256, 'tr'),
         ('256_bl', 256, 'bl'), ('256_br', 256, 'br')]
OFFSETS = (STRIDE // 2, 0)   # 16프레임 창의 가운데 프레임(prepare_cctv_dataset.py 방식) 또는 첫 프레임


def preprocess(frame_bgr, crop):
    """BGR 프레임 -> (CLIP 입력 [3,224,224] float32, 원본 프레임에서 본 영역 (x0,y0,x1,y1))."""
    import cv2
    name, short, pos = crop
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if short is None:
        img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        region = (0, 0, w, h)
    else:
        s = short / min(h, w)
        nh, nw = max(224, round(h * s)), max(224, round(w * s))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        y0 = {'c': (nh - 224) // 2, 'tl': 0, 'tr': 0, 'bl': nh - 224, 'br': nh - 224}[pos]
        x0 = {'c': (nw - 224) // 2, 'tl': 0, 'tr': nw - 224, 'bl': 0, 'br': nw - 224}[pos]
        img = img[y0:y0 + 224, x0:x0 + 224]
        region = (x0 / s, y0 / s, (x0 + 224) / s, (y0 + 224) / s)
    x = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(x).permute(2, 0, 1), region


def read_frames(video_path, indices):
    """필요한 프레임 번호만 순차 디코딩으로 읽는다 (seek는 mp4에서 부정확할 수 있음)."""
    import cv2
    want = set(int(i) for i in indices)
    out, cap, i = {}, cv2.VideoCapture(video_path), 0
    last = max(want) if want else -1
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    cap.release()
    return out


class ModelGradCAM:
    def __init__(self, model, prompt_text, device, use_personalization=False, visual_length=256):
        self.model = model
        self.prompt_text = prompt_text
        self.device = device
        self.use_personalization = use_personalization
        self.visual_length = visual_length
        # 기울기가 fp16에서 사라지지 않도록 영상 인코더만 float32 복사본으로 계산
        self.visual = copy.deepcopy(model.clipmodel.visual).float().to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.crop, self.offset = CROPS[0], OFFSETS[0]

    # ---- CLIP 영상 인코더 (VisionTransformer.forward와 동일, 마지막 블록 입력을 꺼냄) ----
    def _encode(self, x, grad=False):
        v = self.visual
        x = v.conv1(x.to(self.device))
        b, w = x.shape[:2]
        x = x.reshape(b, w, -1).permute(0, 2, 1)
        cls = v.class_embedding + torch.zeros(b, 1, w, device=x.device)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding
        x = v.ln_pre(x).permute(1, 0, 2)
        blocks = v.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        acts = x                      # [197, B, 768]: Grad-CAM 대상 (마지막 블록 입력)
        if grad:                      # 여기서부터 기울기를 추적 (CLIP 가중치는 동결)
            acts = acts.detach().requires_grad_(True)
            x = acts
        x = blocks[-1](x).permute(1, 0, 2)
        feat = v.ln_post(x[:, 0, :]) @ v.proj
        return feat, acts

    @torch.no_grad()
    def calibrate(self, video_path, stored_feat, snippets):
        """저장 특징과 가장 비슷해지는 (자르기, 오프셋)을 고른다. 반환: 평균 코사인 유사도."""
        snippets = [t for t in snippets if t < len(stored_feat)][:3]
        if not snippets:
            return None
        frames = read_frames(video_path, [t * STRIDE + o for t in snippets for o in OFFSETS])
        best = (-2.0, self.crop, self.offset)
        for crop in CROPS:
            for off in OFFSETS:
                xs, ref = [], []
                for t in snippets:
                    f = frames.get(t * STRIDE + off)
                    if f is not None:
                        xs.append(preprocess(f, crop)[0])
                        ref.append(stored_feat[t])
                if not xs:
                    continue
                live = self._encode(torch.stack(xs))[0].float().cpu()
                cos = F.cosine_similarity(live, torch.tensor(np.array(ref), dtype=torch.float32)).mean().item()
                if cos > best[0]:
                    best = (cos, crop, off)
        _, self.crop, self.offset = best
        return best[0]

    def frame_index(self, t):
        return t * STRIDE + self.offset

    def explain(self, frame_bgr, visual, length, t):
        """visual: 데이터셋이 주는 저장 특징 텐서 (process_split 결과), t: 설명할 스니펫 번호.
        반환: (프레임 크기 히트맵 0~1, 스니펫 t 이상 점수, 저장 특징과의 코사인 유사도)"""
        import cv2
        x, region = preprocess(frame_bgr, self.crop)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            feat, acts = self._encode(x[None], grad=True)
            vis = visual.to(self.device).float()
            if vis.dim() == 2:                       # 길이 < 256: [256, 512] -> [1, 256, 512]
                vis = vis.unsqueeze(0)
            stored = vis[t // self.visual_length, t % self.visual_length].detach().clone()
            vis = vis.clone()
            vis[t // self.visual_length, t % self.visual_length] = feat[0].to(vis.dtype)
            from scripts.localization_demo import split_lengths
            logits = self.model(vis, self.prompt_text, split_lengths(int(length)),
                                use_personalization=self.use_personalization)
            logits = logits.reshape(-1, logits.shape[-1])
            score = 1 - logits[t].float().softmax(-1)[0]
            score.backward()
        a = acts.detach()[1:, 0, :].float()           # [196, 768]
        g = acts.grad[1:, 0, :].float()
        cam = F.relu(a @ g.mean(0))                   # Grad-CAM: 채널 가중치 = 공간 평균 기울기
        if cam.max() <= 0:                            # 기울기가 모두 음수면 크기 기준으로 대체
            cam = (a * g).sum(-1).abs()
        g_ = int(round(cam.numel() ** 0.5))
        cam = cam.reshape(g_, g_).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in region]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        heat = np.zeros((h, w), np.float32)
        if x1 > x0 and y1 > y0:
            heat[y0:y1, x0:x1] = cv2.resize(cam, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        cos = F.cosine_similarity(feat[0].detach().float().cpu()[None],
                                  stored.float().cpu()[None]).item()
        return np.clip(heat, 0, 1), float(score.detach()), cos


def blend_cam(frame_bgr, heat, alpha=0.5, floor=0.2):
    import cv2
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (np.clip((heat - floor) / (1 - floor), 0, 1) * alpha)[..., None]
    return (frame_bgr * (1 - mask) + color * mask).astype(np.uint8)