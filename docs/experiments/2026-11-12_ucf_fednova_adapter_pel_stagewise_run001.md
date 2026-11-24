# UCF-Crime 연합 이상 행동 탐지에서 집계 규칙과 의미 정렬 손실의 단계별 적용 효과

- 날짜: 실행 2026-11-11 ~ 2026-11-12 / 보고 2026-11-12
- 코드 커밋: 미기록 (`manifest.json`의 `git_revision`이 null, 작업 디렉터리가 Git 저장소가 아님)
- 상태: 동작 확인 완료 / 성능은 단일 시드 관측 (성능 확인 미완)
- 관련 결정 기록: [0004 집계 트랙 결합](../decisions/0004-aggregation-track-integration.md), [0005 집계 가중치 단위](../decisions/0005-aggregation-weight-unit.md)

## 1. 실험 목적과 가설

Fed-WSVAD(Federated Weakly Supervised Video Anomaly Detection) 기본 구성에서 출발하여 두 가지 기법을 순차로 적용하고, 각 단계가 탐지 성능과 연합학습 내부 거동에 미치는 영향을 분리해 관측한다.

| 단계 | 구성 | 적용 목적 |
| --- | --- | --- |
| S1 | Fed-WSVAD 기본 구성 (FedAvg) | 기준선 |
| S2 | S1 + FedNova 집계 + 저랭크 개인화 프롬프트 어댑터 | 클라이언트 간 로컬 스텝 수 불균형이 만드는 목적 불일치(objective inconsistency) 완화 |
| S3 | S2 + 고정 전체 클래스 의미 정렬 손실 (fixed-bank PEL) | 모든 클라이언트를 동일한 고정 의미 기준점으로 정렬하여 피처 공간에서의 클라이언트 발산 완화 |

### 1.1 S2를 적용한 이유

Fed-WSVAD의 UCF-Crime 클라이언트 분할은 클라이언트마다 보유 데이터량이 크게 다르다. 학습 루프는 정상 배치 한 개와 이상 배치 한 개를 짝지어 처리하므로, epoch당 optimizer 스텝 수는 두 DataLoader 길이의 최솟값으로 결정된다. 본 실험에서 실측한 클라이언트별 총 로컬 스텝 수(`tau`, 15라운드 중 1라운드 기준 10 epoch 합계)는 다음과 같다.

- event 분할 (클라이언트 13개): `[70, 60, 60, 70, 130, 40, 70, 190, 220, 40, 40, 140, 60]` — 최대/최소 비 5.5배
- scene 분할 (클라이언트 9개): `[410, 280, 40, 70, 90, 40, 40, 10, 10]` — 최대/최소 비 **41배**

FedAvg는 각 클라이언트의 학습 종료 지점을 데이터량 가중으로 평균한다. 로컬 스텝을 많이 수행한 클라이언트는 전역 모델에서 더 멀리 이동하므로, 데이터량 가중치와 별개로 평균을 추가로 끌어당긴다. FedNova(Wang et al., NeurIPS 2020)는 각 클라이언트의 누적 업데이트를 자신의 스텝 수로 나눈 뒤 평균하고, 가중 평균 스텝 수로 재조정하여 이 편향을 제거한다. 저랭크 개인화 프롬프트 어댑터는 클라이언트별 프롬프트 보정항 `Delta_P_i = (alpha / r) * A_i @ B_i`를 각 클라이언트에 영구 보존하여 집계에서 제외한다.

### 1.2 S3을 적용한 이유

연합 환경에서 클라이언트는 각자의 데이터 분포에 맞추어 학습하므로 서로 다른 방향으로 이동하고, 평균 과정에서 상충하는 성분이 상쇄된다. 고정 전체 클래스 의미 정렬 손실은 동결된 CLIP 텍스트 인코더로 전체 클래스 프롬프트를 한 번 인코딩해 만든 고정 앵커 집합을 모든 클라이언트에 동일하게 배포하고, 각 영상의 전경 풀링 벡터를 자기 클래스 앵커로, 배경 풀링 벡터를 정상 클래스 앵커로 끌어당긴다. 앵커는 학습되지 않으며 클라이언트 간 동일하므로, 파라미터 공간이 아닌 피처 공간에서 클라이언트를 같은 의미 공간으로 수렴시키는 것을 의도한다.

### 1.3 가설과 성공 기준

- 가설 1: S2는 로컬 스텝 수 편차가 큰 분할에서 S1 대비 ROC-AUC를 개선한다. scene 분할에서 event 분할보다 큰 개선을 보일 것이다.
- 가설 2: S3은 S2 대비 클라이언트 업데이트의 상호 상쇄를 줄이고 탐지 성능을 추가로 개선한다.
- 관측 지표: ROC-AUC, Average Precision, 라운드별 업데이트 보존 비율(`retention_ratio`), 분류 손실과 의미 정렬 손실의 gradient 크기비 및 방향 일치도.

## 2. 통제 조건

### 2.1 공통 조건 (여섯 실행 전부 동일)

| 항목 | 값 |
| --- | --- |
| 데이터셋 / 특징 | UCF-Crime, VadCLIP 배포 CLIP ViT-B/16 특징, 차원 512 |
| 분할 | event (클라이언트 13개), scene (클라이언트 9개) |
| 평가 | 전역 공통 테스트셋, 프레임 단위, `use_personalization=False` |
| 전역 라운드 / 로컬 epoch | 15 / 10 |
| 배치 크기 | 정상 64 + 이상 64 |
| 학습률 / optimizer | 1e-5 / AdamW |
| 시드 | 88888888 (단일) |
| `visual_layers` / `visual_length` | 2 / 256 |
| 집계 가중치 단위 | `loader_batches` (= 정상 DataLoader 배치 수 + 이상 DataLoader 배치 수) |
| 학습 가능 파라미터 | 10,687,652 (S1) / 10,689,716 (S2, S3) |
| 장치 / 라이브러리 | NVIDIA GeForce RTX 3090, CUDA 12.1, PyTorch 2.1.0+cu121 |

클라이언트별 집계 가중치는 세 단계에서 동일하다. event 분할 `[14, 13, 12, 14, 26, 8, 14, 38, 44, 8, 8, 28, 13]`, scene 분할 `[97, 60, 21, 20, 20, 10, 9, 4, 2]`.

### 2.2 단계별 변경 요인

| 항목 | S1 | S2 | S3 |
| --- | --- | --- | --- |
| `aggregation_method` | `fedavg` | `fednova` | `fednova` |
| `server_optimizer` | `none` | `none` | `none` |
| `personalization_type` | `none` | `low_rank` (rank 4, alpha 1.0) | `low_rank` (rank 4, alpha 1.0) |
| `local_param_policy` | `none` | `adapter` | `adapter` |
| `use_pel4vad_contrastive` | 미적용 | 미적용 | 적용 |
| `pel4vad_variant` | 해당 없음 | 해당 없음 | `fixed_bank` |
| `pel4vad_weight` / `pel4vad_temperature` | 해당 없음 | 해당 없음 | 0.3 / 0.09 |
| 실제 서버 클래스 | `FedAvgServer` | `FedStrategyServer` | `FedStrategyServer` |

S1은 기존 Fed-WSVAD 집계 경로(`FedAvgServer`)를 그대로 사용한다. S2와 S3은 집계 규칙과 공유/로컬 정책을 선택할 수 있는 `FedStrategyServer`를 사용하며, 집계 수식 자체는 변경하지 않았다.

고정 의미 앵커 뱅크는 서버가 라운드 0에 한 번 구성하여 모든 클라이언트에 동일 텐서로 배포한다. 인코더는 동결 CLIP ViT-B/16 `encode_text_original`, 풀링은 정규화된 프롬프트 임베딩의 평균 후 재정규화, 지문은 SHA-256 `824c5b6c64385bcf5ad9a55cfd1359c08f802d106e2fb4b1e7d733f06ed76fa6`이다.

### 2.3 짝 비교 전제 확인

S2와 S3 사이에서 클라이언트별 로컬 스텝 수, 집계 가중치, 클라이언트별 클래스 사용 횟수가 전부 일치함을 `round_metrics.jsonl`에서 확인했다. 두 실행의 차이는 의미 정렬 손실의 적용 여부뿐이다.

## 3. 결과

### 3.1 탐지 성능

| 분할 | 구성 | 최고 ROC-AUC | 독립 최고 AP | 최종 ROC-AUC | 최종 AP | 실행 디렉터리 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| event | S1 | 0.8483 | 0.2498 | 0.8472 | 0.2498 | `save/ucf-2026-11-12-16-52-14-470539` |
| event | S2 | 0.8503 | 0.2458 | 0.8485 | 0.2400 | `save/ucf-2026-11-11-17-07-32-100326` |
| event | S3 | 0.8508 | 0.2802 | 0.8405 | 0.2773 | `save/ucf-2026-11-11-20-50-39-870303` |
| scene | S1 | 0.8399 | 0.2406 | 0.8366 | 0.2368 | `save/ucf-2026-11-12-18-45-41-672894` |
| scene | S2 | 0.8505 | 0.2702 | 0.8505 | 0.2592 | `save/ucf-2026-11-11-19-08-05-279499` |
| scene | S3 | 0.8607 | 0.2930 | 0.8607 | 0.2896 | `save/ucf-2026-11-11-22-51-18-319500` |

체크포인트 선택 지표는 UCF-Crime 규약에 따라 ROC-AUC이다. `독립 최고 AP`는 ROC-AUC 최고 라운드의 AP가 아니라 15개 라운드 중 AP 최댓값이며, 선택 대상이 아니었던 지표이다.

### 3.2 단계별 증분 (백분율 포인트)

| 분할 | 단계 | ROC-AUC | AP |
| --- | --- | ---: | ---: |
| event | S1 → S2 | +0.21 | **-0.40** |
| event | S2 → S3 | +0.04 | +3.44 |
| event | S1 → S3 | +0.25 | +3.04 |
| scene | S1 → S2 | +1.06 | +2.95 |
| scene | S2 → S3 | +1.02 | +2.28 |
| scene | S1 → S3 | **+2.08** | **+5.24** |

### 3.3 기존 보고치와의 대조

집계 트랙이 별도로 보고한 UCF-Crime 수치와 본 실험의 S1, S2를 대조했다.

| 항목 | 집계 트랙 보고치 | 본 실험 |
| --- | ---: | ---: |
| S1, event | 84.83 | 84.83 |
| S1, scene | 83.99 | 83.99 |
| S2 − S1, event | +0.21 | +0.21 |
| S2 − S1, scene | +1.06 | +1.06 |

절대값과 증분이 모두 소수점 둘째 자리까지 일치한다. 집계 트랙 저장 디렉터리에는 라운드별 지표 파일이 없어 해당 실행의 AP는 복원할 수 없었으므로, 본 보고서의 AP는 전부 이번 재실행에서 산출한 값이다.

### 3.4 업데이트 보존 비율

`retention_ratio`는 클라이언트별 업데이트를 집계 가중치로 합산한 벡터의 크기를, 각 업데이트 크기의 가중합으로 나눈 값이다. 1에 가까울수록 클라이언트 업데이트가 서로 같은 방향을 향하여 평균 과정에서 덜 상쇄된다. 집계 규칙과 무관하게 동일한 배치 수 가중치로 계산하므로 세 단계를 직접 비교할 수 있다.

| 분할 | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| event | 0.357 | 0.389 | 0.422 |
| scene | 0.450 | 0.639 | 0.672 |

여섯 실행 모두 단계가 올라갈수록 값이 증가한다.

### 3.5 분류 손실과 의미 정렬 손실의 gradient 관계

라운드 1, 5, 10, 15의 첫 배치에서 두 손실의 gradient를 각각 측정했다. 대상은 temporal transformer와 frame position embedding이며, 측정은 읽기 전용으로 optimizer에 기록하지 않는다. 값은 클라이언트 평균이다.

| 분할 | 라운드 | 분류 손실 gradient 크기 | 의미 정렬 손실 gradient 크기 | 가중 적용 후 크기비 | 방향 일치도 (cosine) |
| --- | ---: | ---: | ---: | ---: | ---: |
| event | 1 | 47.50 | 7.50 | 0.048 | 0.517 |
| event | 5 | 43.51 | 4.48 | 0.031 | 0.529 |
| event | 10 | 49.35 | 7.19 | 0.044 | 0.537 |
| event | 15 | 42.02 | 6.32 | 0.046 | 0.568 |
| scene | 1 | 42.15 | 6.81 | 0.049 | 0.547 |
| scene | 5 | 39.53 | 9.05 | 0.070 | 0.536 |
| scene | 10 | 60.18 | 7.08 | 0.035 | 0.619 |
| scene | 15 | 51.46 | 15.49 | 0.104 | 0.476 |

모든 측정 지점에서 방향 일치도가 양수이며, 가중 적용 후 의미 정렬 손실의 기여는 분류 손실의 3퍼센트에서 10퍼센트 범위이다.

### 3.6 연산 비용

파라미터 수는 모델 구성에서 직접 집계했고, 연산량은 `torch.utils.flop_counter.FlopCounterMode`로 추론 경로를 측정했으며, 처리량은 UCF-Crime 테스트 클립 120개(스니펫 24,479개)에 대한 실측이다.

| 항목 | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| 전체 파라미터 | 160,308,389 | 160,310,453 | 160,310,453 |
| 학습 가능 파라미터 | 10,687,652 | 10,689,716 | 10,689,716 |
| 라운드당 집계 대상 파라미터 | 10,687,652 | 10,687,652 | 10,687,652 |
| 추론 연산량 (클립 1개, 스니펫 256개) | 170.413 GFLOPs | 170.413 GFLOPs | 170.413 GFLOPs |
| 추론 처리량 | 46.6 clip/s | 46.9 clip/s | 46.9 clip/s |

- 저랭크 개인화 프롬프트 어댑터는 `A(4 x 4) + B(4 x 512) = 2,064`개를 추가한다. 학습 가능 파라미터 대비 0.019퍼센트이다.
- FedNova는 서버 측 집계 수식만 변경하므로 파라미터를 추가하지 않는다.
- 고정 의미 정렬 손실은 학습 손실이므로 추론 경로에 존재하지 않으며 파라미터와 연산량을 추가하지 않는다. 학습 중에만 크기 28 KB의 고정 앵커 텐서를 보유한다.
- 어댑터가 추가하는 추론 연산량은 16,384 FLOPs로 전체의 0.00001퍼센트이다.
- 어댑터는 클라이언트에 영구 보존되어 집계 대상에서 제외되므로 라운드당 통신량은 세 단계가 동일하다.

추론 연산량의 구성은 다음과 같다.

| 부분 | GFLOPs | 비중 |
| --- | ---: | ---: |
| 텍스트 경로 (동결 CLIP 텍스트 인코더 + 프롬프트 생성기) | 166.92 | 98.0퍼센트 |
| temporal transformer | 3.49 | 2.0퍼센트 |

위 처리량은 CLIP 특징이 사전 추출되어 있다는 전제의 값이다. 동결 CLIP ViT-B/16 영상 인코더를 별도로 측정한 결과 프레임당 35.13 GFLOPs, 처리량 394 frame/s였다. 스니펫당 16프레임 기준으로 특징 추출이 이상 탐지 헤드의 약 844배이므로, 종단간 실시간 처리 상한은 약 394 frame/s이며 세 단계의 기여는 측정 한계 내에서 0이다.

### 3.7 학습 소요 시간

| 분할 | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| event | 113분 | 119분 | 119분 |
| scene | 92분 | 101분 | 99분 |

S2는 S1 대비 약 7퍼센트 길다. S3은 S2와 차이가 없으며 scene에서는 오히려 2분 짧게 측정되었다.

## 4. 해석

### 4.1 가설 1 — 지지됨

S2는 scene 분할에서 ROC-AUC를 1.06포인트, event 분할에서 0.21포인트 개선했다. 로컬 스텝 수 편차가 41배인 scene 분할에서 5.5배인 event 분할보다 다섯 배 큰 개선을 보였으며, 이는 FedNova가 해결 대상으로 삼는 불균형의 크기와 방향이 일치한다. 업데이트 보존 비율도 scene에서 0.450에서 0.639로 크게 올랐고 event에서는 0.357에서 0.389로 소폭 올라 같은 순서를 따른다.

### 4.2 가설 2 — 부분 지지

업데이트 보존 비율은 두 분할 모두에서 S2 대비 증가했고(event 0.389 → 0.422, scene 0.639 → 0.672), 이는 고정 의미 앵커가 클라이언트 발산을 줄인다는 설계 의도와 일치한다. 성능 측면에서는 분할에 따라 양상이 다르다.

- scene 분할: ROC-AUC 1.02포인트, AP 2.28포인트 개선. 두 지표 모두 일관된 개선이다.
- event 분할: ROC-AUC 0.04포인트로 사실상 변화 없음. AP는 3.44포인트 개선.

의미 정렬 손실은 전체 순위 품질(ROC-AUC)보다 이상 구간 지목의 정밀도(AP)를 개선하는 방향으로 작용한다. 다만 AP는 체크포인트 선택 지표가 아니므로, 보고한 AP는 15개 라운드 중 최댓값이며 ROC-AUC 기준으로 선택된 체크포인트의 AP와 다르다.

### 4.3 event 분할에서 S1 → S2의 AP 하락

event 분할에서 S2는 S1 대비 ROC-AUC를 0.21포인트 올리면서 AP를 0.40포인트 낮췄다. 이 방향 불일치는 설명되지 않았다. 단일 시드 관측이므로 실재 여부부터 반복 실험으로 확인해야 한다.

### 4.4 두 기법의 상호작용

scene 분할에서 S1 → S2 → S3의 증분이 ROC-AUC 기준 +1.06, +1.02로 거의 같은 크기로 누적되었고, AP 기준으로도 +2.95, +2.28로 누적되었다. 업데이트 보존 비율도 세 단계에서 단조 증가했다. 두 기법이 각각 집계 규칙과 피처 공간이라는 서로 다른 층위에서 작용하며 효과가 상쇄되지 않음을 시사한다.

### 4.5 의미 정렬 손실의 가중치 여유

가중 적용 후 의미 정렬 손실의 gradient 기여가 분류 손실의 3퍼센트에서 10퍼센트에 머물고 방향 일치도가 모든 측정 지점에서 양수였다. 두 손실이 경쟁하지 않으며 가중치 0.3이 과하지 않음을 뜻한다. 다만 scene 분할 라운드 15에서 기여가 0.104로 상승하고 방향 일치도가 0.476으로 최저를 기록했으므로, 가중치를 올릴 경우 후반 라운드의 방향 일치도를 함께 관측해야 한다.

## 5. 주장 가능 범위와 미검증 항목

### 동작 확인 (완료)

- S1, S2, S3 세 구성이 각각 의도한 서버 경로와 설정으로 15라운드를 완주했다.
- S2와 S3의 짝 비교 전제(로컬 스텝 수, 집계 가중치, 클래스 사용 횟수 일치)를 기계적으로 확인했다.
- 여섯 실행 모두 라운드별 지표와 진단 기록이 15행씩 생성되었고 `scripts/analyze_pel_diagnostics.py` 검증을 통과했다.
- S1, S2가 집계 트랙 기존 보고치를 소수점 둘째 자리까지 재현했다.

### 성능 유효성 (미완)

- **모든 수치가 시드 88888888 단일 실행이다.** 라운드 간 ROC-AUC 변동폭이 약 1포인트이므로, event 분할의 S2 → S3 변화(+0.04)와 S1 → S2의 AP 변화(-0.40)는 이 변동폭 안에 들어간다. 반복 없이는 판단할 수 없다.
- scene 분할의 S1 → S3 변화(+2.08)는 변동폭보다 크지만 역시 단일 관측이다.

### 수렴 상태

scene 분할에서 S1은 라운드 8에 최고값(0.8399)에 도달한 뒤 완만히 하락하여 15라운드 예산이 충분했다. 반면 S2와 S3은 최종 라운드인 15에서 최고값을 기록했으므로 수렴 전일 가능성이 있다. 15라운드 고정 예산은 집계 트랙 프로토콜을 따른 것이며, S2와 S3의 보고치는 해당 예산 안에서의 값으로 읽어야 한다.

### 그 밖의 미검증 항목

- random 분할과 XD-Violence 데이터셋은 실행하지 않았다.
- 집계 트랙에만 존재하는 ConceptNet 기반 지식 프롬프트(`prompt_source`)는 본 코드베이스에 없으므로 비교에 포함되지 않았다.
- 체크포인트 선택을 AP 기준으로 바꿨을 때의 수치는 산출하지 않았다.
- 개인화 추론 모드(클라이언트 어댑터 적용) 성능은 전역 테스트셋 대리 지표로만 기록되어 있으며 본 보고서의 비교 대상이 아니다.
- 학습 중 최대 GPU 메모리 사용량은 측정하지 않았다.

## 6. 후속 실험

1. 시드 2개 이상 추가 실행. 현재 단일 관측이므로 가장 우선순위가 높다. 특히 event 분할의 AP 하락과 S2 → S3 ROC-AUC 변화의 실재 여부를 확인한다.
2. 의미 정렬 손실 가중치 0.5 이상 탐색. 4.5의 gradient 관측이 근거이며 후반 라운드 방향 일치도를 함께 기록한다.
3. scene 분할 라운드 수 연장. S2와 S3의 수렴 여부를 확인한다.
4. random 분할과 XD-Violence로 확장하여 집계 트랙 보고표와 대응시킨다.

## 7. 트랙 경계와 호환성

- 본 실험은 집계 수식, 집계 가중치, 서버 optimizer, 공유/로컬 파라미터 정책을 변경하지 않았다. `fed/aggregation.py`는 집계 트랙 저장소와 바이트 단위로 동일하다.
- 집계 트랙 설정을 사용하기 위해 `--pel_allow_aggregation_integration` 명시적 선택지를 통해 세 축을 개방했으며, 근거는 결정 기록 0004에 있다.
- 집계 가중치 단위의 기존 불일치(가중치는 두 DataLoader 길이의 합, 실제 스텝은 최솟값)는 결정 기록 0005에서 선택지로 분리했고, 본 실험은 기존 단위 `loader_batches`를 사용하여 기존 기록과의 비교 가능성을 유지했다.

## 8. 재현 명령

```
# S1 event
python -u train.py --dataset ucf --batch_size 64 --learning_rate 1e-05 --global_rounds 15 \
  --local_epochs 10 --server_optimizer none --cuda 0 --aggregation_method fedavg \
  --split_mode event --clients_num 13 --seed 88888888 --personalization_type none \
  --local_param_policy none --pel_diagnostics

# S2 event
python -u train.py --dataset ucf --batch_size 64 --learning_rate 1e-05 --global_rounds 15 \
  --local_epochs 10 --server_optimizer none --cuda 0 --aggregation_method fednova \
  --split_mode event --clients_num 13 --seed 88888888 --personalization_type low_rank \
  --local_param_policy adapter --pel_allow_aggregation_integration --pel_diagnostics

# S3 event
python -u train.py --dataset ucf --batch_size 64 --learning_rate 1e-05 --global_rounds 15 \
  --local_epochs 10 --server_optimizer none --cuda 0 --aggregation_method fednova \
  --split_mode event --clients_num 13 --seed 88888888 --personalization_type low_rank \
  --local_param_policy adapter --pel_allow_aggregation_integration \
  --use_pel4vad_contrastive --pel4vad_variant fixed_bank --pel4vad_temperature 0.09 \
  --pel4vad_weight 0.3 --pel_diagnostics
```

scene 분할은 `--split_mode scene --clients_num 9`로 바꾼다. 명령 매니페스트는 `artifacts/experiment-commands/2026-11-11_ucf_fednova_adapter_pel_aggregation_protocol.json`과 `artifacts/experiment-commands/2026-11-12_ucf_fedavg_baseline_aggregation_protocol.json`에 있고, 학습 로그는 `logs/fednova_adapter_pel/`과 `logs/fedavg_baseline/`에 있다.

<!-- experiment-results
{
  "schema_version": 1,
  "experiment": "ucf_fednova_adapter_pel_stagewise",
  "run": 1,
  "status": "completed",
  "report_date": "2026-11-12",
  "seeds": [88888888],
  "sample_size": 1,
  "checkpoint_selection": "highest test ROC-AUC round",
  "conditions": {
    "dataset": "ucf",
    "evaluation": "global frame-level common test set",
    "batch_size_per_loader": 64,
    "learning_rate": 1e-05,
    "global_rounds": 15,
    "local_epochs": 10,
    "optimizer": "AdamW",
    "visual_layers": 2,
    "visual_length": 256,
    "aggregation_weight_unit": "loader_batches",
    "device": "NVIDIA GeForce RTX 3090",
    "torch_version": "2.1.0+cu121",
    "git_commit": null,
    "pel_bank_sha256": "824c5b6c64385bcf5ad9a55cfd1359c08f802d106e2fb4b1e7d733f06ed76fa6"
  },
  "results": [
    {"stage": "S1", "split": "event", "clients_num": 13, "aggregation_method": "fedavg", "personalization_type": "none", "use_pel4vad_contrastive": false, "best_roc": 0.8483, "independent_best_ap": 0.2498, "final_roc": 0.8472, "final_ap": 0.2498, "retention_ratio_mean": 0.357, "train_minutes": 113, "run_directory": "save/ucf-2026-11-12-16-52-14-470539"},
    {"stage": "S2", "split": "event", "clients_num": 13, "aggregation_method": "fednova", "personalization_type": "low_rank", "use_pel4vad_contrastive": false, "best_roc": 0.8503, "independent_best_ap": 0.2458, "final_roc": 0.8485, "final_ap": 0.2400, "retention_ratio_mean": 0.389, "train_minutes": 119, "run_directory": "save/ucf-2026-11-11-17-07-32-100326"},
    {"stage": "S3", "split": "event", "clients_num": 13, "aggregation_method": "fednova", "personalization_type": "low_rank", "use_pel4vad_contrastive": true, "pel4vad_weight": 0.3, "best_roc": 0.8508, "independent_best_ap": 0.2802, "final_roc": 0.8405, "final_ap": 0.2773, "retention_ratio_mean": 0.422, "train_minutes": 119, "run_directory": "save/ucf-2026-11-11-20-50-39-870303"},
    {"stage": "S1", "split": "scene", "clients_num": 9, "aggregation_method": "fedavg", "personalization_type": "none", "use_pel4vad_contrastive": false, "best_roc": 0.8399, "independent_best_ap": 0.2406, "final_roc": 0.8366, "final_ap": 0.2368, "retention_ratio_mean": 0.450, "train_minutes": 92, "run_directory": "save/ucf-2026-11-12-18-45-41-672894"},
    {"stage": "S2", "split": "scene", "clients_num": 9, "aggregation_method": "fednova", "personalization_type": "low_rank", "use_pel4vad_contrastive": false, "best_roc": 0.8505, "independent_best_ap": 0.2702, "final_roc": 0.8505, "final_ap": 0.2592, "retention_ratio_mean": 0.639, "train_minutes": 101, "run_directory": "save/ucf-2026-11-11-19-08-05-279499"},
    {"stage": "S3", "split": "scene", "clients_num": 9, "aggregation_method": "fednova", "personalization_type": "low_rank", "use_pel4vad_contrastive": true, "pel4vad_weight": 0.3, "best_roc": 0.8607, "independent_best_ap": 0.2930, "final_roc": 0.8607, "final_ap": 0.2896, "retention_ratio_mean": 0.672, "train_minutes": 99, "run_directory": "save/ucf-2026-11-11-22-51-18-319500"}
  ],
  "cost": {
    "total_params": {"S1": 160308389, "S2": 160310453, "S3": 160310453},
    "trainable_params": {"S1": 10687652, "S2": 10689716, "S3": 10689716},
    "aggregated_params_per_round": {"S1": 10687652, "S2": 10687652, "S3": 10687652},
    "inference_gflops_per_clip_256_snippets": {"S1": 170.413, "S2": 170.413, "S3": 170.413},
    "inference_clips_per_second": {"S1": 46.6, "S2": 46.9, "S3": 46.9},
    "clip_visual_encoder_gflops_per_frame": 35.13,
    "clip_visual_encoder_frames_per_second": 394,
    "peak_gpu_memory": null
  }
}
-->
