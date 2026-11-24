# 코드 구조

손실 설명에는 [논문 용어와 코드 식별자 규약](loss-terminology.md)을 적용한다. 문서의 기본 검출 손실 표현은 MIL 기반 분류 손실이다.

| 위치 | 역할 |
| --- | --- |
| train.py | 학습 진입점 |
| inference.py | 추론 진입점 |
| fed/ | 연합학습 클라이언트와 서버 |
| utils/config.py | CLI 인자와 기본 설정 |
| utils/model.py | 모델 구현 |
| utils/dataset.py | 데이터셋 구현 |
| scripts/ | 실험 및 데이터 준비 |
| logs/, runs_4090/ | 기존 실험 산출물 |

## 모델과 학습 데이터 흐름

1. `utils/dataset.py`가 사전 추출 CLIP 특징을 읽어 학습 시 256개 위치로 리샘플링 또는 padding한다.
2. `Model.encode_video()`가 위치 임베딩과 Temporal Transformer로 시각 특징을 만든다.
3. `PromptLearner`가 이벤트 텍스트와 배치 시각 문맥을 결합해 4개 context token을 만든다.
4. 동결된 CLIP text encoder가 context와 이벤트 이름을 인코딩한다.
5. 정규화한 시각·텍스트 특징의 내적과 CLIP logit scale로 클래스 logits을 만든다.
6. MIL 기반 분류 손실은 유효 길이 내 클래스별 top-k logits으로 영상 단위 감독을 적용한다.
7. 추론 점수는 `1 - P(Normal)`이며 각 snippet 점수를 16개 프레임으로 반복한다.

`PromptTranslator`는 영상별 문맥을 배치 차원에서 평균하므로 생성 프롬프트는 현재 배치의 구성에도 의존한다. `lengths`는 MIL 기반 분류 손실에서 사용되지만 모델의 시간 attention과 프롬프트 생성에는 전달되지 않는다.

## 파라미터 공유 정책

| 경로 | 공유 및 로컬 정책 |
| --- | --- |
| `FedAvgServer`와 `FedAvgClient` | 모든 학습 가능한 파라미터를 집계 |
| `FedPGPServer`와 `FedStrategyClient` | 공유 파라미터를 집계하고 개인화 adapter는 클라이언트에 보존 |
| `FedStrategyServer` | 모델의 local policy에 따라 adapter와 LayerNorm을 로컬에 둘 수 있음 |

CLIP 파라미터는 동결한다. 개인화 adapter를 제외한 Temporal Transformer, 위치 임베딩과 Prompt Generator는 기본적으로 공유된다. 글로벌 평가는 개인화 adapter를 비활성화한다. 현재 집계 가중치는 클라이언트가 반환하는 DataLoader 배치 수이며 원시 샘플 수와 정확히 같지 않다.

## 상세 분석

- [PEL 적용 분석과 이벤트 확장 판단](pel4vad-integration.md): 적용 위치, 손실, 로그, 구현 문제와 방법론 후보.
- [현행 성능 목표](../README.md#현행-연구-목표와-성능-정의)
- [집계 방식](../../AGGREGATION.md)
