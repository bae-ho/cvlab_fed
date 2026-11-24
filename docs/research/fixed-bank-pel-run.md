# 고정 전체 클래스 PEL 실행

현재 1단계 재실행 조건과 Linux 명령은 [손실·업데이트 진단 명세](pel-stage1-diagnostic-spec.md)를 따른다. 네 조건을 한 번에 자동 실행하지 않고 한 조건씩 실행한다.

2026-10-06 구현. 기본 Fed-WSVAD의 MIL 기반 분류 손실과 FedAvg를 유지하고 PEL4VAD의 전경·배경 의미 정렬을 결합한 변형이다. 원본 전체 재현 또는 SOTA 달성 결과가 아니다. 성능 목표는 [현행 정의](../README.md#현행-연구-목표와-성능-정의)를 따른다.

## 구현

2026-10-08 후속 코드: [5단계 계획](../plans/active/2026-10-08-pel-federated-compatibility.md), [세미나 핵심 기록](../seminar/2026-10-08_PEL_연합학습_5단계_실험구현.md). 아래 2026-10-06 기본 후보의 동작은 새 옵션을 전달하지 않으면 유지한다. 이전 4개 실제 실행은 [결과 기록](2026-10-08-ucf-event-scene-results.md)에 있으며, 새 후보의 실제 데이터 학습은 미실행이다.

- `utils/fixed_pel.py`: 동결 CLIP의 `encode_text_original`로 고정 기준 생성. 전체 클래스 KL과 유효 길이 pooling 구현.
- 모든 클라이언트가 UCF 14개/CCTV 7개 기준을 공유한다. 배치에 없는 클래스도 denominator에 포함된다.
- PEL 선택 점수와 텍스트 기준은 detach한다. PEL은 Temporal/위치 임베딩을 학습하며 MIL 기반 분류 손실은 기존 프롬프트 학습을 유지한다.
- 한 클라이언트의 여러 단일 라벨 이벤트 영상에 적용 가능하다. 한 영상의 다중 라벨 및 XD는 지원하지 않는다. 분할과 sampler는 변경하지 않았다.
- 기존 `legacy` 손실을 보존하고 기본 FedAvg의 PEL 인자 누락을 직접 수정했다. 기본값은 PEL off다.
- 집계 가중치는 기존 DataLoader 배치 수이며 표본 수로 변경하지 않았다. checkpoint parameter key/shape와 추론 경로도 동일하다.
- 새 pooling만 패딩을 제외한다. 기존 Temporal/프롬프트 내부의 패딩 처리는 유지한다.

### 선택적 후속 옵션

| 옵션 | 역할 | 기본값 |
| --- | --- | --- |
| `--pel_diagnostics` | 손실 gradient·client delta 진단; plain FedAvg UCF/CCTV 전용 | off |
| `--pel_warmup_rounds` / `--pel_ramp_rounds` | 초기 PEL 비활성 / 선형 계수 증가 | 0 / 0 |
| `--pel_pooling` | score 선택 또는 유효 snippet 균등 pooling | score |
| `--pel_class_balance` | PEL 정렬 target row의 batch 클래스 역빈도 기여 | none |
| `--pel_gradient_scope` | PEL gradient 전체 또는 마지막 Temporal 블록 제한 | all |
| `--pel_projection` | identity 초기화·공유 PEL 전용 head; checkpoint에 새 key 추가 | off |
| `--pel_allow_aggregation_integration` | 기존 FedNova/FedPGP와 fixed-bank의 명시적 연결; 집계 식 유지 | off |

후속 옵션은 enabled fixed_bank 조건에서만 사용한다. projection 후보 checkpoint 추론은 동일 `--pel_projection`을 전달한다. head는 검출 forward에서 미사용이며 PEL을 통해 H를 학습한다. `scripts/prepare_pel_experiments.py`는 명령 JSON만 생성하며 학습을 실행하지 않는다. 반복 seed 결과는 `scripts/summarize_pel_runs.py`로 기존 실행 디렉토리를 읽어 취합한다. 코드·torch·서버·비PEL 조건이 다른 실행은 자동으로 같은 paired 비교에 합치지 않는다.

## 비교 실행

데이터·CLIP 가중치·GPU 준비 후 같은 seed, 초기 checkpoint, 분할 및 예산으로 각각 실행한다. 아래 기본 예산은 20 round × 10 local epoch이며 이번 작업에서 실제 학습하지 않았다.

```powershell
python train.py --dataset ucf --split_mode event --clients_num 13 --aggregation_method fedavg --server_optimizer none --personalization_type none --local_param_policy none --seed 88888888
python train.py --dataset ucf --split_mode event --clients_num 13 --aggregation_method fedavg --server_optimizer none --personalization_type none --local_param_policy none --seed 88888888 --use_pel4vad_contrastive --pel4vad_variant fixed_bank --pel4vad_weight 1.0 --pel4vad_temperature 0.09
```

UCF scene은 `--split_mode scene --clients_num 9`를 양쪽에 동일 적용한다. CCTV는 `--dataset cctv --split_mode event --clients_num 6`이다. 클라이언트 수와 생성 loader 수가 다르면 오류를 반환한다. 새로운 혼합 분할을 생성하지 않았다.

선택적인 `--pel4vad_prompts path.json`은 canonical dataset key마다 비어 있지 않은 문자열 리스트를 요구한다. UCF의 경우 `Normal`, `Abuse`, ..., `Vandalism` 모두가 필요하다. 각 문구 임베딩을 정규화한 후 클래스별 평균을 다시 정규화한다. 처음 비교는 옵션 없이 기존 클래스 문구를 사용하고, 문구 확대는 별도 비교 조건으로 기록한다.

## 산출물과 검증

`save/<dataset-timestamp>/manifest.json`은 설정·seed·실제 서버/클라이언트·코드 revision/dirty·torch·bank 문구와 checksum을 기록한다. `round_metrics.jsonl`은 매 round ROC/AP, 클라이언트 MIL 기반 분류 손실/PEL/total/step/사용 클래스 횟수, 집계 배치 수를 기록한다. `model.pth`는 기존처럼 test의 최고 round, `final_model.pth`는 마지막 round다. `summary.json`은 best/final 지표와 선택 정책을 기록한다. 기존 test에서 best를 선택하는 정책은 재현용이며 validation 선택 검증을 완료한 것으로 해석하지 않는다.

```powershell
python -m unittest discover -s tests -v
python train.py --help
```

CPU 합성 검증은 gradient 경계, 전체 denominator, padding, 극단 점수, 잘못된 라벨/길이, bank RNG/모드 보존, 옵션 off 업데이트 일치, 혼합 이벤트 학습, 2-client/2-round 집계 및 저장을 확인한다. 실제 데이터 성능·GPU 실행·Fed-WSVAD 대비 향상은 아직 측정하지 않았다.

2026-10-06 검증 환경: Python 3.13.5, PyTorch 2.10.0+cu126, CPU. 누락된 ftfy/einops를 설치했다. 자동 검증 7개 통과, `train.py --help` 및 `git diff --check` 통과. 실제 Model 경로 검증은 CLIP만 작은 대역 모델로 교체하여 다운로드 없이 실행했다. summary에는 최고 checkpoint의 ROC/AP와 독립 최고 AP도 별도로 기록한다.
