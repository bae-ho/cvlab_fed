"""카메라 실시간 이상 탐지 데모 (젯슨 나노 + USB 웹캠/CSI 카메라, 영상 파일로도 테스트 가능).

동작:
  1) 카메라 프레임을 계속 받아 화면에 표시하고,
  2) 16프레임마다 한 장을 학습 때와 같은 방식으로 잘라 CLIP ViT-B/16 특징을 만들고,
  3) 최근 --window 개 스니펫 특징으로 학습된 모델을 돌려 '가장 최근 스니펫'의 이상 점수와 종류를 계산해
  4) 화면 위쪽 배너(정상/ANOMALY)와 아래쪽 점수 그래프로 보여준다.
CPU에서는 2)~3)이 0.53초(16프레임)보다 오래 걸리므로, 계산이 끝날 때마다 가장 최신 프레임을 가져가는
방식으로 동작한다 (밀린 스니펫은 건너뜀 → 화면은 매끄럽고 판정은 몇 초마다 갱신).

CSI 카메라는 pip OpenCV로 열 수 없어서(GStreamer 미포함) gst-launch-1.0을 띄워 원시 프레임을 파이프로 읽는다.

사용 예 (젯슨, 저장소 루트):
    # USB 웹캠(OBSBOT Meet 2 등) + 젯슨 모니터에 표시
    python scripts/live_demo.py --checkpoint save/edge/model.pth --personalization_type low_rank --source 0

    # CSI 카메라
    python scripts/live_demo.py --checkpoint save/edge/model.pth --personalization_type low_rank --source csi \
        --width 640 --height 360

    # 영상 파일을 카메라처럼 실시간 재생하며 테스트 (모니터 없이 결과 영상 저장)
    python scripts/live_demo.py --checkpoint save/edge/model.pth --personalization_type low_rank \
        --source ~/ucf_videos/Arrest024_x264.mp4 --realtime --no_window --save artifacts/live_test.mp4
"""
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

STRIDE = 16


# ----------------------------------------------------------------------------
# 프레임 소스
# ----------------------------------------------------------------------------
class RawPipeSource:
    """BGR 원시 프레임을 stdout으로 내보내는 명령(gst-launch, ffmpeg 등)에서 프레임을 읽는다."""

    def __init__(self, cmd, width, height, fps):
        self.w, self.h, self.fps = width, height, fps
        self.nbytes = width * height * 3
        self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.nbytes * 4)

    def read(self):
        buf = self.proc.stdout.read(self.nbytes)
        if buf is None or len(buf) < self.nbytes:
            err = self.proc.stderr.read().decode(errors='ignore')[-800:] if self.proc.poll() is not None else ''
            if err:
                print('[카메라 프로세스 메시지]\n' + err)
            return False, None
        return True, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()

    def release(self):
        try:
            self.proc.kill()
        except Exception:
            pass


class CvSource:
    def __init__(self, src, realtime, width=None, height=None, fps=None, mjpg=True):
        import cv2
        if str(src).isdigit():   # USB(UVC) 카메라: V4L2로 열고 MJPG + 해상도 지정 (USB 2.0 대역폭 때문)
            self.cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
            if mjpg:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            if width and height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 지연 최소화
        else:
            self.cap = cv2.VideoCapture(os.path.expanduser(src))
        if not self.cap.isOpened():
            raise SystemExit(f'소스를 열 수 없습니다: {src}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if str(src).isdigit():
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            print(f'USB 카메라: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} '
                  f'{self.fps:.0f}fps {fourcc.to_bytes(4, "little").decode(errors="ignore")}')
        self.realtime = realtime and not str(src).isdigit()
        self.t0, self.n = None, 0

    def read(self):
        ok, f = self.cap.read()
        if ok and self.realtime:   # 파일을 실제 카메라 속도로 재생
            if self.t0 is None:
                self.t0 = time.time()
            self.n += 1
            wait = self.t0 + self.n / self.fps - time.time()
            if wait > 0:
                time.sleep(wait)
        return ok, f

    def release(self):
        self.cap.release()


def csi_command(sensor_id, cap_w, cap_h, fps, out_w, out_h, flip):
    caps = f'video/x-raw(memory:NVMM),width={cap_w},height={cap_h},framerate={fps}/1'
    return (f"gst-launch-1.0 -q nvarguscamerasrc sensor-id={sensor_id} ! {shlex.quote(caps)} ! "
            f"nvvidconv flip-method={flip} ! video/x-raw,width={out_w},height={out_h},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! fdsink fd=1 sync=false")


def open_source(args):
    if args.source == 'csi':
        cmd = csi_command(args.sensor_id, args.cap_width, args.cap_height, args.fps,
                          args.width, args.height, args.flip)
        print('카메라 명령:', cmd)
        return RawPipeSource(cmd, args.width, args.height, args.fps)
    if args.source.startswith('cmd:'):
        return RawPipeSource(args.source[4:], args.width, args.height, args.fps)
    return CvSource(args.source, args.realtime, args.width, args.height, args.fps, not args.no_mjpg)


# ----------------------------------------------------------------------------
# 모델 쪽
# ----------------------------------------------------------------------------
def cache_constant_text(model):
    """encode_text_prompt가 매번 다시 계산하는 '클래스 이름 문장' 특징은 입력과 무관한 상수이므로 캐시한다.
    (결과는 같고, 텍스트 인코더 계산이 절반으로 줄어든다)"""
    orig = model.clipmodel.encode_text_original
    cache = {}

    def cached(tokens):
        key = (tuple(tokens.shape), int(tokens.sum()))
        if key not in cache:
            cache[key] = orig(tokens)
        return cache[key]

    model.clipmodel.encode_text_original = cached


def truncate_text_encoder(model):
    """프롬프트 문장 인코딩을 문장 끝(EOT) 토큰까지만 계산한다.
    CLIP 텍스트 트랜스포머는 앞쪽만 보는(causal) 구조이고 EOT 위치의 출력만 쓰므로,
    EOT 뒤의 빈 토큰(77개 중 대부분)을 잘라내도 결과는 같다 (실측 오차 ~1e-6). 계산량은 수 배 감소."""
    clip = model.clipmodel
    full_mask = clip.build_attention_mask()

    def encode_text(text, token):
        L = int(token.argmax(dim=-1).max()) + 1
        x = text[:, :L].type(clip.dtype) + clip.positional_embedding[:L].type(clip.dtype)
        x = x.permute(1, 0, 2)
        blocks = clip.transformer.resblocks
        saved = [b.attn_mask for b in blocks]
        try:
            for b in blocks:
                b.attn_mask = full_mask[:L, :L]
            x = clip.transformer(x)
        finally:
            for b, m in zip(blocks, saved):
                b.attn_mask = m
        x = clip.ln_final(x.permute(1, 0, 2)).type(clip.dtype)
        return x[torch_arange(x.shape[0], x.device), token.argmax(dim=-1)] @ clip.text_projection

    clip.encode_text = encode_text


def torch_arange(n, device):
    import torch
    return torch.arange(n, device=device)


class LiveScorer:
    def __init__(self, model, prompt_text, device, crop, window, use_personalization, visual_length=256):
        import torch
        from scripts.gradcam import CROPS
        self.torch = torch
        self.model, self.prompt_text, self.device = model, prompt_text, device
        self.crop = {c[0]: c for c in CROPS}[crop]
        self.feats = deque(maxlen=window)
        self.use_personalization = use_personalization
        self.visual_length = visual_length

    def encode(self, frame_bgr):
        from scripts.gradcam import preprocess
        torch = self.torch
        x, _ = preprocess(frame_bgr, self.crop)
        clip = self.model.clipmodel
        with torch.inference_mode():
            f = clip.encode_image(x[None].to(self.device, dtype=clip.visual.conv1.weight.dtype))
        self.feats.append(f[0].float().cpu().numpy())

    def score(self):
        """반환: (모든 위치 점수 배열, 가장 최근 스니펫 점수, 최근 스니펫 이상 클래스 이름)"""
        torch = self.torch
        from scripts.localization_demo import split_lengths
        from utils.tools import process_split
        feats = np.stack(self.feats).astype(np.float32)
        n = len(feats)
        vis, _ = process_split(feats, self.visual_length)
        vis = torch.from_numpy(np.ascontiguousarray(vis)).to(self.device)
        if vis.dim() == 2:
            vis = vis.unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(vis, self.prompt_text, split_lengths(n), use_personalization=self.use_personalization)
            probs = logits.reshape(-1, logits.shape[-1])[:n].float().softmax(-1).cpu().numpy()
        scores = 1 - probs[:, 0]
        cls = self.prompt_text[int(probs[-1, 1:].argmax()) + 1]
        return scores, float(scores[-1]), cls


# ----------------------------------------------------------------------------
# 화면 구성
# ----------------------------------------------------------------------------
def compose(frame, st, threshold, disp_w, hist_len=120):
    import cv2
    h, w = frame.shape[:2]
    disp_h = int(h * disp_w / w)
    img = cv2.resize(frame, (disp_w, disp_h))
    score, ema, cls = st.get('score'), st.get('ema'), st.get('cls', '')
    anomalous = ema is not None and ema >= threshold
    font = cv2.FONT_HERSHEY_SIMPLEX
    if ema is None:
        color, text = (90, 90, 90), st.get('status', 'warming up...')
    elif anomalous:
        color, text = (40, 40, 220), f'ANOMALY: {cls}  {ema:.2f}'
    else:
        color, text = (60, 160, 60), f'normal  {ema:.2f}'
    cv2.rectangle(img, (0, 0), (disp_w, 40), color, -1)
    cv2.putText(img, text, (10, 28), font, 0.8, (255, 255, 255), 2)
    if anomalous:
        cv2.rectangle(img, (0, 0), (disp_w - 1, disp_h - 1), color, 5)

    # 아래쪽: 최근 판정 점수 그래프 + 처리 속도
    strip_h = 110
    strip = np.full((strip_h, disp_w, 3), 255, np.uint8)
    hist = list(st.get('hist', []))[-hist_len:]
    ty = int(strip_h - 20 - threshold * (strip_h - 30))
    cv2.line(strip, (0, ty), (disp_w - 1, ty), (160, 160, 160), 1)
    if len(hist) >= 2:
        xs = np.linspace(0, disp_w - 1, hist_len)[-len(hist):].astype(int)
        ys = (strip_h - 20 - np.clip(hist, 0, 1) * (strip_h - 30)).astype(int)
        cv2.polylines(strip, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (156, 78, 31), 2)
    info = (f"update {st.get('period', 0):.1f}s | CLIP {st.get('enc_ms', 0):.0f}ms | model {st.get('model_ms', 0):.0f}ms"
            f" | window {st.get('window', 0)} | skipped {st.get('skipped', 0)} | cam {st.get('cam_fps', 0):.0f}fps")
    cv2.putText(strip, info, (8, strip_h - 5), font, 0.45, (60, 60, 60), 1)
    return np.vstack([img, strip])


# ----------------------------------------------------------------------------
def build_parser():
    from utils import config
    p = config.parser
    g = p.add_argument_group('live demo')
    g.add_argument('--source', default='csi',
                   help="USB 카메라 번호(0, OBSBOT 등 웹캠) | csi | 영상 파일 경로 | 'cmd:<BGR 원시 프레임 명령>'")
    g.add_argument('--sensor_id', type=int, default=0)
    g.add_argument('--cap_width', type=int, default=1280)
    g.add_argument('--cap_height', type=int, default=720)
    g.add_argument('--width', type=int, default=1280, help='프레임 너비 (USB 카메라 요청 해상도, csi/cmd 출력 크기)')
    g.add_argument('--height', type=int, default=720)
    g.add_argument('--no_mjpg', action='store_true', help='USB 카메라에서 MJPG 대신 기본 형식 사용')
    g.add_argument('--fps', type=int, default=30)
    g.add_argument('--flip', type=int, default=0, help='CSI 영상 회전 (0=그대로, 2=180도)')
    g.add_argument('--realtime', action='store_true', help='영상 파일을 실제 속도로 재생 (카메라 흉내)')
    g.add_argument('--no_drop', action='store_true',
                   help='스니펫을 건너뛰지 않고 전부 처리 (파일 검증용. 카메라에서는 화면이 밀림)')
    g.add_argument('--window', type=int, default=64, help='모델에 넣을 최근 스니펫 수 (최대 256)')
    g.add_argument('--crop', default='256_center',
                   help='특징 추출 때의 자르기 방식. UCF 체크포인트는 256_center (Grad-CAM 보정 결과)')
    g.add_argument('--offset', type=int, default=0, help='16프레임 창에서 쓸 프레임 위치 (UCF: 0)')
    g.add_argument('--threshold', type=float, default=0.5)
    g.add_argument('--ema', type=float, default=0.5, help='점수 지수평활 계수 (1=평활 없음)')
    g.add_argument('--display_width', type=int, default=960)
    g.add_argument('--no_window', action='store_true', help='화면 표시 안 함 (SSH 실행, --save와 함께)')
    g.add_argument('--save', default=None, help='표시 화면을 mp4로 저장')
    g.add_argument('--log', default='artifacts/live_log.csv')
    g.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help='torch 계산 스레드 수. 화면·카메라용으로 코어 1개를 남겨야 함 (전부 쓰면 크게 느려짐)')
    g.add_argument('--max_seconds', type=float, default=None)
    g.add_argument('--no_fast_text', action='store_true', help='텍스트 인코더 EOT 절단 최적화 끄기 (비교용)')
    return p


def main():
    import cv2
    import torch

    from fed.server import CCTV_LABEL_MAP, UCF_LABEL_MAP, XD_LABEL_MAP
    from scripts.localization_demo import load_model
    from utils.tools import get_prompt_text

    args = build_parser().parse_args()
    if args.checkpoint is None:
        raise SystemExit('--checkpoint 를 지정하세요')
    torch.set_num_threads(args.threads)
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    label_map = {'ucf': UCF_LABEL_MAP, 'xd': XD_LABEL_MAP, 'cctv': CCTV_LABEL_MAP}[args.dataset]
    prompt_text = get_prompt_text(dict(label_map))

    print(f'모델 로드 중 (device={device})...')
    t0 = time.time()
    model, use_personalization = load_model(args, device)
    cache_constant_text(model)
    if not args.no_fast_text:
        truncate_text_encoder(model)
    scorer = LiveScorer(model, prompt_text, device, args.crop, min(args.window, 256), use_personalization,
                        args.visual_length)
    print(f'모델 준비 {time.time() - t0:.0f}s')

    src = open_source(args)
    jobs = queue.Queue(maxsize=1)
    st = {'hist': deque(maxlen=500), 'status': 'warming up (first result in a few seconds)...', 'skipped': 0}
    lock = threading.Lock()
    stop = threading.Event()
    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
    log_f = open(args.log, 'w', newline='', encoding='utf-8')
    log = csv.writer(log_f)
    log.writerow(['time', 'frame', 'window', 'score', 'ema', 'class', 'encode_ms', 'model_ms'])

    def worker():
        last_done = None
        try:
            while not stop.is_set():
                try:
                    fidx, frame = jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                if fidx is None:
                    break
                t1 = time.time()
                scorer.encode(frame)
                t2 = time.time()
                _, score, cls = scorer.score()
                t3 = time.time()
                with lock:
                    ema = score if st.get('ema') is None else args.ema * score + (1 - args.ema) * st['ema']
                    st.update(score=score, ema=ema, cls=cls, enc_ms=(t2 - t1) * 1000, model_ms=(t3 - t2) * 1000,
                              window=len(scorer.feats), period=(t3 - last_done) if last_done else (t3 - t1))
                    st['hist'].append(ema)
                last_done = t3
                log.writerow([f'{t3:.3f}', fidx, len(scorer.feats), f'{score:.4f}', f'{ema:.4f}', cls,
                              f'{(t2 - t1) * 1000:.0f}', f'{(t3 - t2) * 1000:.0f}'])
                log_f.flush()
        except Exception as e:  # 작업 스레드 오류를 화면 루프에 알림
            import traceback
            traceback.print_exc()
            st['error'] = repr(e)
            stop.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    writer, win = None, 'Anomaly detection (q: quit)'
    fidx, t_start, cam_t = 0, time.time(), deque(maxlen=30)
    try:
        while not stop.is_set():
            ok, frame = src.read()
            if not ok:
                print('프레임을 더 받을 수 없습니다 (영상 끝 또는 카메라 오류).')
                break
            cam_t.append(time.time())
            if fidx % STRIDE == args.offset:
                if args.no_drop:
                    jobs.put((fidx, frame))
                else:
                    try:
                        jobs.put_nowait((fidx, frame))
                    except queue.Full:            # 이전 작업이 아직 안 끝남 → 최신 프레임으로 교체
                        try:
                            jobs.get_nowait()
                        except queue.Empty:
                            pass
                        jobs.put_nowait((fidx, frame))
                        st['skipped'] += 1
            with lock:
                if len(cam_t) > 1:
                    st['cam_fps'] = (len(cam_t) - 1) / max(cam_t[-1] - cam_t[0], 1e-6)
                view = compose(frame, st, args.threshold, args.display_width)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'),
                                             getattr(src, 'fps', 30) or 30, (view.shape[1], view.shape[0]))
                writer.write(view)
            if not args.no_window:
                cv2.imshow(win, view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            fidx += 1
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
    finally:
        if args.no_drop:          # 파일 검증 모드: 남은 작업을 끝까지 처리
            jobs.put((None, None))
            th.join()
        stop.set()
        # 계산 중이던 판정이 끝날 때까지 기다린 뒤 종료 (안 기다리면 torch 스레드가 남아
        # 'terminate called without an active exception' 으로 비정상 종료됨)
        th.join(timeout=180)
        src.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        log_f.close()
    if st.get('error'):
        raise SystemExit(f"처리 중 오류: {st['error']}")
    print(f'로그: {args.log}' + (f'  저장 영상: {args.save}' if args.save else ''))


if __name__ == '__main__':
    main()