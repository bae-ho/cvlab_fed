# PEL4VAD 적용 분석과 이벤트 확장 판단

- 작성일: 2026-10-06
- 분석 대상: 현재 작업 디렉터리의 코드와 기존 실험 로그.
- 현행 목표: [연구 목표와 성능 정의](../README.md#현행-연구-목표와-성능-정의).
- 구현 상태: 기존 PEL 변형 구현을 분석했다. 아래 개선 후보는 미구현이다.

PEL 결합은 다중 이벤트 VAD 성능을 높일 수 있는 타당한 후보다. 한 이벤트와 Normal만으로 수행한 로컬 실험은 원본 PEL의 이벤트 간 의미 구분 효과를 충분히 검증하지 못한다. 서로 다른 이벤트의 시각 학습 데이터와 적절한 텍스트 기준을 함께 늘리는 방향을 먼저 검토한다. 성능 향상은 가설이며 로그로 확인된 달성 사실과 구분한다.

## 기반 모델과 분석 범위

이 저장소의 기반은 AAAI 2025 Fed-WSVAD이다. ECCV 2024의 FedVAD와는 다른 방법이다. 재사용하는 주요 구현은 `train.py`, `inference.py`, `fed/`, `utils/`, `scripts/`에 있다. CLIP 구현은 동결된 영상·텍스트 encoder로 사용된다.

| 위치 | 분석 대상 |
| --- | --- |
| [모델](../../utils/model.py) | Temporal 출력, 동적 텍스트 특징, logits와 보조 출력 |
| [프롬프트 생성](../../utils/prompt_net.py) | 시각 문맥의 배치 평균과 cross attention |
| [손실](../../utils/tools.py) | MIL 기반 분류 손실, PEL pooling과 KL, 개인화 대조 손실 |
| [단일 클라이언트 실험](../../scripts/experiment_pel4vad_client.py) | 데이터 필터, 두 클래스 dictionary, 학습 및 전체 테스트 평가 |
| [클라이언트](../../fed/client.py) | 로컬 PEL 추가와 공유 파라미터 반환 |
| [서버](../../fed/server.py) | 전체 라벨 dictionary, 집계와 글로벌 평가 |
| [진입점](../../train.py) | 옵션 소비와 legacy 경로 선택 |
| [데이터 준비](../../scripts/prepare_cctv_dataset.py) | CCTV 라벨, 프레임 특징, 분할과 GT 구성 |

## 기존 적용 위치와 계산

| 단계 | 코드 동작 | 주요 shape |
| --- | --- | --- |
| 시각 인코딩 | CLIP 특징에 위치 임베딩과 Temporal Transformer 적용 | H: B × T × 512 |
| 텍스트 인코딩 | 시각 문맥과 클래스 텍스트로 생성한 context를 동결된 CLIP에 입력 | Z: C × 512 |
| 검출 logits | 정규화한 H와 Z의 내적에 동결된 CLIP scale 적용 | B × T × C |
| 이상 점수 | s = 1 - softmax(logits)[Normal] | B × T |
| 전경 pooling | exp(10s) - 1을 시간축에서 정규화하여 H를 가중 평균 | B × 512 |
| 배경 pooling | exp(10(1-s)) - 1을 정규화하여 H를 가중 평균 | B × 512 |
| PEL 목표 | 배치에서 만든 영상·텍스트 쌍의 cosine 행렬과 동일 클래스 목표에 KL 적용 | M × M |

이상 영상은 전경과 해당 이벤트 텍스트, 배경과 Normal 텍스트의 두 쌍을 만든다. 정상 영상은 높은 이상 점수로 pooling한 특징과 Normal 텍스트의 한 쌍을 만든다. 높은 이상 점수의 정상 구간을 정상으로 감독하는 것은 원본에도 있는 hard-negative 처리다.

전체 손실은 `MIL 기반 분류 손실 + pel4vad_weight * PEL`이다. 현재 PEL temperature 기본값은 0.09이고 weight 기본값은 1.0이다. 설정 기본값의 원본은 [utils/config.py](../../utils/config.py)다.

### Gradient가 흐르는 위치

| 대상 | 현재 PEL gradient |
| --- | --- |
| Temporal Transformer와 위치 임베딩 | 흐름 |
| 동적 Prompt Generator | 흐름 |
| 활성화된 개인화 adapter | 흐름 |
| pooling에 사용한 예측 logits | 흐름. detach 없음 |
| CLIP encoder 파라미터 | 동결되어 업데이트 없음 |

CLIP 동결과 입력 프롬프트 gradient 차단은 다르다. 학습 가능한 context를 통과하는 text encoder 연산은 프롬프트까지 미분된다.

### 두 실행 경로의 차이

| 항목 | 단일 클라이언트 스크립트 | 연합학습의 PEL 경로 |
| --- | --- | --- |
| 데이터 | Normal과 지정 이벤트 하나 | 클라이언트 분할별 데이터 |
| 모델 시작점 | 새 모델 초기화. 연합 checkpoint 로드 없음 | 글로벌 모델을 기반으로 로컬 업데이트 |
| 텍스트 dictionary | Normal과 지정 이벤트 두 항목 | 서버가 제공하는 전체 데이터셋 클래스 |
| 개인화 | none | 설정에 따라 adapter 적용 |
| 서버 집계 | 없음 | 공유 파라미터 집계 |
| 평가 | 학습하지 않은 클래스까지 포함한 전체 테스트 | 전체 이벤트의 글로벌 테스트 |

단일 실험은 PEL을 붙인 로컬 모델의 동작 검증이다. 연합학습 효과의 직접 검증이 아니며, 그 실험의 성능 저하만으로 연합 PEL을 기각하지 않는다.

## 실제 사전 크기와 대조 후보의 차이

확장할 대상을 세 가지로 구분한다.

| 대상 | 의미 | 예시 |
| --- | --- | --- |
| 이벤트 클래스 수 | 실제 시각 데이터의 라벨 종류 | UCF 13개 이상 이벤트와 Normal |
| 클래스별 의미 표현 수 | 클래스에 연결된 설명과 지식 개념의 수 | fighting에 연결된 행동 및 관계 개념 |
| 실제 대조 후보 수 | 손실의 denominator에 들어가는 텍스트와 클래스 수 | 현재 배치에서 선택된 텍스트 쌍 |

원본 PEL의 UCF 학습은 13개 이상 이벤트와 Normal을 사용한다. XD는 6개 이상 이벤트와 Normal이다. 따라서 모든 데이터셋에 동일한 이벤트 수를 적용하는 방식이 아니다. 클래스별 지식 개념은 CLIP으로 인코딩한 뒤 평균하여 하나의 의미 특징으로 만든다. 지식 개념을 늘리는 것이 클래스 수를 늘리는 것은 아니다. [PEL 논문](https://arxiv.org/pdf/2306.14451), [공식 데이터셋](https://github.com/yujiangpu20/PEL4VAD/blob/master/dataset.py)

공식 `dataset.py`는 각 영상에 Normal과 해당 이벤트의 텍스트 두 개를 제공한다. `get_cas()`는 여러 영상의 쌍을 배치에서 결합하고 `create_logits()`가 대조 행렬을 만든다. 배치에 여러 이벤트가 있어야 서로 다른 이벤트의 텍스트가 함께 들어온다. 공식 구현이 매번 전체 사전의 모든 클래스를 denominator에 넣는 것은 아니다. [공식 pooling과 logits](https://github.com/yujiangpu20/PEL4VAD/blob/master/utils.py)

현재 코드도 `token_feats.append(text_features[cls])`와 Normal만 선택한다. 따라서 다음을 구분해야 한다.

| 변경 | 현재 구현에서 발생하는 효과 | 판단 |
| --- | --- | --- |
| Fight 영상만 유지하고 dictionary에 다른 이벤트 이름 추가 | 동적 프롬프트와 검출 softmax는 변경될 수 있음. 다른 이벤트 텍스트가 PEL에 자동으로 모두 포함되지는 않음 | 다중 이벤트 PEL 검증으로 불충분 |
| 각 이벤트의 시각 데이터와 대응 텍스트 추가 | 배치에 등장한 이벤트가 PEL 후보에 포함됨 | 원본 조건에 가까워져 개선 가능성 있음 |
| 같은 dictionary의 각 클래스에 지식 기반 의미 특징 제공 | 고정된 의미 기준의 정보가 풍부해짐 | 이름만 사용하는 현재 방식보다 유효할 가능성 있음 |
| 전체 클래스의 고유 텍스트를 항상 denominator에 사용 | 데이터가 없는 클래스도 명시적인 음성 후보로 사용 가능 | 손실 수정이 필요한 별도 변형. 원본 재현과 구분 |

사전에 새 클래스를 추가할 때 문자열 라벨, `label_map`, `prompt_text`, `sample_class_idx`를 함께 맞춰야 한다. 기존 단일 실험의 `sample_class_idx = ones(...)`는 이벤트 하나가 index 1인 경우만 처리하므로 다중 이벤트 확장 시 라벨별 인덱스로 바꿔야 한다.

### 연합학습에서의 추가 조건

UCF event split은 이미 전체 시스템에서 13개 이벤트를 사용하지만 각 클라이언트의 이상 데이터는 한 이벤트다. 클라이언트마다 여전히 한 이벤트만 학습하면 로컬 PEL의 이벤트 간 대조는 충분히 생기지 않는다. 서버 파라미터 평균은 원본의 다중 이벤트 배치를 그대로 대체하지 않는다.

- 단일 클라이언트에 다중 이벤트를 제공하면 PEL 자체의 의미 구분 효과를 확인할 수 있다.
- scene 또는 random split은 클라이언트에 여러 이벤트가 있을 수 있지만 실제 배치의 클래스 구성도 확인해야 한다.
- event split을 유지하려면 전체 고정 텍스트 기준을 사용하는 손실 변형 등을 검토할 수 있다.
- 데이터 분할을 바꾸는 경우 기준 모델도 같은 분할로 비교해야 한다. 쉬운 분할에서 오른 성능을 방법론 효과로 혼동하지 않는다.

### 성능 향상 가능성에 대한 결론

**다중 이벤트의 실제 시각 데이터, 대응 텍스트, 원본의 의미 기준을 함께 늘리는 접근은 성능이 올라갈 합리적인 가능성이 있다.** 원본 PEL이 제공하는 이벤트 간 의미 감독을 한 이벤트 실험에서는 활용하기 어렵기 때문이다. 이에 대한 적절한 다음 판단은 원본 조건에 가까운 적용을 검토하는 것이며, 즉시 PEL을 폐기하거나 모든 복합 모듈을 추가하는 것이 아니다.

다만 이벤트 수만 맞춰도 향상이 보장되는 것은 아니다. 라벨 혼동, 배치 내 클래스 편중, 움직이는 텍스트 기준과 padding 문제가 남는다. 사전 크기와 클래스 수의 증가는 구현 경로 확인 뒤 평가해야 한다. 상세 실행 명령, seed 및 학습 예산은 후속 실험 설계에서 확정한다.

## 기존 로그의 관찰

아래 차이는 기록된 소수 네 자리 값으로 계산했다. 단위는 퍼센트포인트이며 서로 다른 실행의 완전한 설정 일치를 보증하지 않는다.

| CCTV 단일 학습 이벤트 | seed 수 | 평균 AUC 변화 | 평균 AP 변화 |
| --- | --- | --- | --- |
| Fight | 4 | +2.00 | +1.18 |
| Flood | 4 | -3.22 | -0.60 |
| Crowd | 4 | -0.46 | -0.36 |
| Invasion | 4 | -3.90 | -1.02 |
| Gathering | 4 | -1.34 | -1.04 |
| Falldown | 4 | -2.04 | -0.35 |

- 근거: [기본 CCTV 로그](../../logs/pel4vad_experiments/)와 [seed 확인 로그](../../logs/pel4vad_seed_check/).
- [UCF Fighting 로그](../../logs/pel4vad_ucf/)는 4개 seed 평균 AUC +1.55%p, AP +1.33%p이다.
- CCTV lambda sweep에서도 이벤트와 seed에 따라 결과가 달라지므로 weight 축소만으로 개선이 보장되지 않는다. [sweep 로그](../../logs/pel4vad_lambda_sweep/)

| CCTV 연합 실행 로그 | 기록된 최고 AUC |
| --- | --- |
| [FedAvg](../../logs/cctv_fedavg_train.log) | 60.22% |
| [FedPGP](../../logs/cctv_fedpgp_train.log) | 61.39% |
| [FedPGP와 PEL](../../logs/cctv_fedpgp_pel4vad_train.log) | 59.51% |

이 표는 `FedAvgServer` 또는 `FedPGPServer`로 6개 CCTV 클라이언트를 20 round 학습한 별도 연합 로그다. 앞의 standalone 단일 이벤트 스크립트는 집계가 없으며 MIL 기반 분류 손실 대 PEL만 비교한다. 두 결과를 단일 클라이언트의 FedAvg 대 FedPGP 효과로 해석하지 않는다. [새 기본 FedAvg PEL 계획](../plans/active/2026-10-06-fedavg-fixed-anchor-pel.md)

현재 서버는 매 round의 test 지표로 최고 checkpoint를 선택한다. 위 수치는 기존 기록이며, 현행 성공 기준에 맞는 validation 선택 기반의 새 비교 결과와 구분한다.

## 확인된 구현 문제와 원인 가설

| 구분 | 관찰 | 영향 및 후속 판단 |
| --- | --- | --- |
| 코드 사실 | 기본 legacy FedAvg 경로가 PEL flag를 서버에 전달하지 않음 | PEL을 켰다는 CLI만으로 실제 활성화를 판단하면 안 됨 |
| 코드 사실 | PEL 함수에 lengths가 없고 모델 attention에도 길이 mask가 없음 | 위치 임베딩 등을 거친 padding 특징이 pooling과 문맥에 포함될 수 있음 |
| 코드 사실 | PromptTranslator가 배치 시각 문맥을 평균 | 학습 배치와 테스트 영상의 프롬프트 조건이 다름 |
| 코드 사실 | 이상 영상마다 배경을 Normal로 감독 | 정상 배경이 거의 없는 영상에도 동일한 목표가 적용됨 |
| 원본과의 차이 | 현재 텍스트 기준은 동적 학습 프롬프트, 원본은 사전 추출 지식 특징 | 고정 의미 기준의 효과를 검증할 필요 있음 |
| 원인 가설 | 시각 특징과 동적 텍스트가 함께 이동하여 특정 로컬 클래스에 적응 | 다른 클래스 성능 하락의 가능한 원인. 로그만으로 확정하지 않음 |
| 원인 가설 | 초기 오분류가 pooling과 의미 정렬에 재사용 | warm-up과 선택 점수 detach 비교 후보 |
| 구성 효과 | 정상 텍스트가 반복되어 denominator의 클래스별 개수가 다름 | 고유 클래스 기준 또는 가중 방식과 비교. 원본에도 유사 구성 있음 |

`attn_window`, `prompt_prefix`, `prompt_postfix`는 현재 forward에서 기대한 조절 기능으로 사용되지 않으며, `global_text_prompt_embeddings`도 현재 forward에서 참조하지 않는다. PromptTranslator의 `Linear(256, 32)`와 추론의 256 고정값 때문에 visual_length 변경은 CLI만으로 해결되지 않는다.

PEL과 별개의 `fedpgp_contrastive_loss()`는 글로벌 텍스트를 handcrafted 기준에 당기고 개인화 텍스트를 상대 음성으로 두는 손실이다. 전경·배경 PEL과 같은 방법으로 취급하지 않는다.

### CCTV 자료의 해석 제한

[전처리 로그](../../logs/cctv_prepare.log)에 학습 238개, 테스트 31개, 정상 테스트 3개가 기록되어 있다. 정상 학습 영상 18개가 모든 이벤트 클라이언트에 복제된다. 이 환경에서의 개선은 자체 데이터 결과이며 공개 연합학습 벤치마크 SOTA의 직접 근거가 아니다. 실제 프레임 라벨과 장면·촬영 세션 중복은 데이터가 있을 때 확인한다.

## 적용 후보와 우선순위

| 순위 | 후보 | 적용 위치와 동작 | 상태 |
| --- | --- | --- | --- |
| 우선 | 다중 이벤트와 원본 의미 기준을 반영한 PEL | Temporal 출력에서 전경·배경 pooling, 동결된 지식 기반 텍스트에 정렬. 기존 MIL 기반 분류 손실 유지 | 후속 검토 |
| 다음 | 별도 semantic projection | Temporal 출력에 작은 projection을 추가하여 의미 정렬과 검출 표현의 간섭을 조절 | 우선 후보 이후 판단 |
| 조건부 | 이진 검출 head와 의미 보조 손실 분리 | Temporal 출력 뒤 이진 MIL head와 PEL 정렬 분리 | 현재 SOTA 목표에 대한 추가 구조 후보. 미학습 검증 요구 때문이 아님 |
| 조건부 | 클라이언트 간 prototype 정렬 | 공통 표현 공간에서 정상·이벤트 prototype을 집계하고 로컬 표현 정렬 | 통신 및 감독 변경을 동반하는 후속 후보 |

고정 의미 기준을 사용하는 경우 기존 동적 프롬프트는 검출 경로에 유지하고 PEL의 기준만 분리할 수 있다. projection만 학습하고 기존 검출 표현에 gradient를 전달하지 않으면 실제 검출 개선으로 연결되지 않을 수 있으므로 연결 방식을 명시해야 한다.

개인화 adapter에만 손실을 적용하면 글로벌 평가에서 adapter를 비활성화하는 현재 경로와 목표가 어긋날 수 있다. 글로벌 성능을 목표로 하는 개선은 공유 표현에도 전달되어야 한다.

prototype 기반 정상 의미 정렬은 [FedVAR](https://arxiv.org/abs/2608.06876) 등 기존 연구와 중복될 수 있다. 방법론 신규성과 성능 SOTA는 별도로 판단한다. 이 후보를 채택하더라도 자동으로 새로운 방법이라고 주장하지 않는다.

## 코드 수준 검증 근거

- 기존 분석에서 `utils.tools.pel4vad_contrastive_loss`를 작은 텐서로 호출하여 시각·텍스트·선택 logits의 gradient가 모두 존재함을 확인했다.
- 동일한 두 개 유효 위치 뒤에 추가 위치를 포함한 결과와 유효 위치만 사용한 결과의 손실이 달랐다. 길이 mask 부재의 동작을 확인한 것이며 실제 데이터에서의 성능 하락량을 측정한 결과는 아니다.
- 현재 디렉터리에는 데이터와 checkpoint가 없어 기존 학습 성능을 재현하지 않았다. 로그 수치는 기존 산출물 요약이다.

## 원문과 후속 참조

- [Fed-WSVAD 논문](https://ojs.aaai.org/index.php/AAAI/article/view/35398)
- [PEL4VAD 논문](https://arxiv.org/pdf/2306.14451)
- [PEL4VAD 공식 구현](https://github.com/yujiangpu20/PEL4VAD)
- [공식 학습](https://github.com/yujiangpu20/PEL4VAD/blob/master/train.py), [공식 모델](https://github.com/yujiangpu20/PEL4VAD/blob/master/model.py)
- [목표 변경 결정](../decisions/0002-vad-performance-objective.md)
- [사전 확장 해석 평가 사례](../../harness/evals/pel4vad-dictionary-coverage.md)

## 두 트랙 및 분할 호환성
[현재 작업 경계](../research/two-track-scope.md)에 따라 집계·서버 optimizer·개인화 정책을 유지한다. 모듈 적용점, 고정 의미 기준, 시각 특징과 손실의 연결을 먼저 검토한다. 집계 개선과 PEL 통합 구현은 이후 각 트랙 검증 뒤 진행한다.

다중 이벤트 단일 로컬 실험은 PEL 진단에 쓰며 event 연합 실험을 대체하지 않는다. event 로컬의 단일 이상 클래스와 scene 로컬의 실제 배치 클래스 구성을 각각 확인한다. 다른 클라이언트 영상 반입이나 scene/event 변경을 PEL의 전제 조건으로 만들지 않는다. 새 모듈의 공유 상태·key/shape 및 gradient 경로를 기록해 추후 집계 트랙과의 통합 충돌을 확인한다.

[Slack S05 대조](../research/slack-experiment-evidence.md): FedPGP 최고 ROC 61.39는 본문 61 미만 요약과 다르다. 원표 AP 64.89는 최고 ROC round AP이며 독립 최고 AP는 65.29다. PEL 후기 ROC 약 50은 로그와 부합하지만 PEL 전체 폐기 근거로 확대하지 않는다.
# 현행 구현 안내

2026-10-06 기본 FedAvg에 `fixed_bank` PEL 변형을 연결했다. [실행 안내](../research/fixed-bank-pel-run.md)를 현재 동작의 기준으로 사용한다. 아래 기존 코드/로그 분석의 동적 텍스트·옵션 누락 설명은 수정 이전 상태이며 legacy 연구 기록으로 보존한다.
