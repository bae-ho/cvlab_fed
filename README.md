# Federated Weakly Supervised video Anomaly Detection with Multimodal Prompt (AAAI2025)

현재 연구 코드·실험 결과·후배 연구원의 기존 PEL 구현과의 차이·남은 작업은 **[인수인계 안내](docs/HANDOVER.md)**부터 확인하세요. 아래 환경 안내는 원본 프로젝트 기준이며 최근 실험 환경은 인수인계 문서에 별도로 기록했습니다.

구현 구조는 [코드 구조](docs/architecture/README.md)와 [PEL 적용 구조 비교](docs/seminar/2026-10-08_event와_scene_분할_및_PEL_적용비교.md)를 참고하세요.

## 🎨 Highlights
* 📹 We propose a novel privacy protected federated learning framework for video anomaly detection, which enables multiple clients to train a unified and generalized video anomaly detector with privacy.
* 🛰️ We design a prompt generator driven by both global and local contexts, which achieves global generalization and local personalization.
* 🧾 We conduct extensive experiments on multiple datasets. In order to simulate real-world scenarios, we re-organize XD-Violence and UCF-Crime datasets. The results demonstrate the superiority of the proposed method.

## 🔰 Get Started
### 🎈 Preparation
* Clone the repository and download depencies. And we use `torch==2.1.0+cu118`.
```bash
git clone https://github.com/wbfwonderful/Fed-WSVAD.git
cd Fed-WSVAD
conda create -n fed-wsvad python=3.11
conda activate fed-wsvad
pip install -r requirements.txt
```
* Download the CLIP features for UCF-Crime and XD-Violence datasets from [VadCLIP](https://github.com/nwpu-zxr/VadCLIP).
* Modify the path in the CSV files in `/data/list` based on the the saving location of the features.

### 🏃 Training
Run the following commands to train the model in different settings and feel free to change the parameters refer to `utils/config.py`.
```python
# for xd event:
python train.py --dataset xd --batch_size 128 --clients_num 6

# for xd random:
python train.py --dataset xd --batch_size 128 --clients_num 10 --split_mode random

# for xd scene:
python train.py --dataset xd --batch_size 128 --clients_num 13 --split_mode scene

# for ucf event:
python train.py

# for ucf random:
python train.py --split_mode random --clients_num 10

# for ucf scene:
python train.py --split_mode scene --clients_num 9

```
### 🧪 Inference
Set `--chechpoint` to specify the path of chechpoints.
```python
# for xd:
python inference.py --dataset xd --checkpoint model.pth
# for ucf:
python inference.py --checkpoint model.path
```

## 🏄‍♂️ References
We referenced the repos below for the code. Thanks for their contributions! 🤗
* [VadCLIP](https://github.com/nwpu-zxr/VadCLIP)
* [FedTPG](https://github.com/boschresearch/FedTPG)
# 고정 전체 클래스 PEL

기본 FedAvg와 MIL 기반 분류 손실에 고정 의미 기준 PEL을 추가한 비교 실행은 [실행 안내](docs/research/fixed-bank-pel-run.md)를 참고한다. 기본 학습은 PEL off이며 새 후보는 `--use_pel4vad_contrastive --pel4vad_variant fixed_bank`로 활성화한다.
