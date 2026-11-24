# 기본 FedAvg와 고정 의미 기준 PEL 수정 계획

- 날짜: 2026-10-06
- 상태: 코드 구현 및 CPU 합성 검증 완료. 2026-10-08 기준 event/scene의 기준·후보 4개 실제 실행 결과를 사용자 로그로 보고받아 기록 완료. 독립 재현·반복성은 미검증.
- 트랙: 김배호 연구원의 PEL 적용 지원.
- 근거: 사용자 요청. [현행 성능 목표](../../README.md#현행-연구-목표와-성능-정의), [트랙 경계](../../research/two-track-scope.md).

## 요청과 성공 기준

Fed-WSVAD 기본 집계를 고정하고 기존 PEL의 전경·배경 정렬을 전체 이벤트에 적용한다. 의미 기준을 고정하여 PEL은 시각 표현에만 gradient를 전달한다. 같은 조건의 Fed-WSVAD 기준 모델과 비교한 뒤 추가 실험 설계를 판단한다.

| 기준 | 구현 및 비교에서 확인할 내용 |
| --- | --- |
| 집계 | 기본 FedAvg 경로, 기존 평균 가중치와 반환 규약 유지 |
| 전체 이벤트 | UCF 13개 이상 이벤트와 Normal, CCTV 6개와 Normal의 전체 텍스트 기준 사용 |
| 의미 기준 고정 | 학습 중 고정 텍스트 값이 변하지 않음 |
| 표현 학습 | PEL 단독 backward에서 시각 경로만 gradient를 받음 |
| 기준과 비교 | 동일 데이터·클라이언트·초기화·예산·추론·checkpoint 선택 정책 |
| 해석 | 실제 새 기준 결과와 비교. 다른 집계 실험 또는 논문 보고치와 혼합하지 않음 |

## 기존 결과의 실행 단위 구분

| 결과 | 실행 단위 | 집계 해석 |
| --- | --- | --- |
| `logs/pel4vad_experiments/`, `pel4vad_seed_check/`, `pel4vad_ucf/` | standalone 로컬 모델. MIL 기반 분류 손실과 PEL 추가 비교 | 서버 없음. FedAvg/FedPGP 비교 결과가 아님 |
| S05의 `cctv_fedavg_train.log`, `cctv_fedpgp_train.log`, `cctv_fedpgp_pel4vad_train.log` | 6개 클라이언트의 20 round CCTV 연합 실행 로그 | 여러 로컬 업데이트를 집계한 별도 결과 |

S05 로그의 최고 ROC는 각각 60.22%, 61.39%, 59.51%다. 최고 round와 해당 AP, final 값은 [Slack 근거](../../research/slack-experiment-evidence.md#s05-cctv-표와-원시-로그)를 따른다. 단일 로컬 성능표에 이 수치를 넣지 않는다. 실행자와 전체 설정을 원시 로그만으로 확정하지 않는다.

클라이언트 하나만 있는 FedAvg는 평균 계수가 1이어서 로컬 업데이트를 그대로 반환한다. FedPGP의 개인화 adapter로 다른 동작이 생길 수는 있지만 그 차이를 여러 클라이언트의 집계 효과로 설명할 수 없다.

## 고정할 조건

| 항목 | 기본 Fed-WSVAD 비교 조건 |
| --- | --- |
| algorithm / 서버·클라이언트 | FedAvg / FedAvgServer / FedAvgClient |
| aggregation_method | fedavg |
| server_optimizer | none |
| personalization_type / local_param_policy | none / none |
| use_fedpgp_contrastive | false |
| 업데이트와 평균 | 현재 FedAvg의 파라미터 반환·배포·DataLoader 배치 수 가중 규약 유지 |
| 클라이언트 데이터 | 기존 event/scene 분할과 참여 규칙 유지. 다른 클라이언트 영상 합침 없음 |
| 검출 모델 | 기존 Temporal과 동적 프롬프트, MIL 기반 분류 손실, 추론의 1-P(Normal) 유지 |

주 비교는 공개 Fed-WSVAD 벤치마크와 직접 연결되는 UCF로 계획한다. event는 전체 13개 이벤트가 이미 연합 시스템에 존재한다. 로컬에 한 이벤트만 있어도 전체 텍스트 기준을 PEL denominator에 넣을 수 있다. scene 비교는 기존 scene 데이터를 그대로 사용하며 별도 결과로 기록한다. 초기 실행에 모든 분할을 자동 추가하지 않는다.

CCTV는 6개 이벤트 데이터를 그대로 사용한다. UCF의 13개 이름을 CCTV에 추가하여 13개 시각 이벤트를 학습한 것으로 보고하지 않는다. XD는 다중 라벨 처리가 추가로 필요하므로 첫 구현 범위에서 제외한다. 다중 이벤트 단일 로컬 진단이 필요하면 별도 조건으로 표시하며 연합 benchmark 점수로 사용하지 않는다.

## 의미 기준과 gradient 설계

### MIL 기반 분류 손실과 원본 PEL4VAD 손실의 관계

Fed-WSVAD는 MIL로 영상 단위 예측을 얻고 분류 손실을 계산한다. 공식 코드의 함수 식별자는 논문 용어와 구분한다. [용어 규약과 원문 근거](../../architecture/loss-terminology.md)를 따른다. 구현은 유효 길이 내 각 클래스의 top-k snippet logits을 평균하고 영상 라벨에 대한 softmax cross entropy를 적용한다. UCF에서는 Normal과 13개 이벤트의 영상 단위 감독을 제공한다.

원본 PEL4VAD의 `CLAS2`는 이진 정상/이상 점수를 사용한다. 정상 영상은 top-1, 이상 영상은 top-k를 선택하고 BCE를 적용한다. 공식 학습은 `CLAS2 + lambda * KLV_loss`로 구성된다. PEL은 기본 검출 손실을 대체하는 것이 아니라 함께 사용하는 의미 정렬 항이다. [공식 분류 손실](https://github.com/yujiangpu20/PEL4VAD/blob/master/loss.py), [BCE criterion 설정](https://github.com/yujiangpu20/PEL4VAD/blob/master/main.py), [공식 학습](https://github.com/yujiangpu20/PEL4VAD/blob/master/train.py)

| 구성 | 검출 감독 | 의미 정렬 |
| --- | --- | --- |
| 기본 Fed-WSVAD | MIL 기반 다중 클래스 분류 손실 | 별도 PEL 없음 |
| 원본 PEL4VAD | CLAS2 이진 MIL BCE | PEL의 KL 정렬 |
| 이번 후보 | 기존 MIL 기반 분류 손실 유지 | 고정 전체 클래스 PEL 변형 추가 |

이번 목적은 PEL 전체 모델의 재현이 아니라 Fed-WSVAD에 PEL을 결합한 성능 비교다. MIL 기반 분류 손실을 원본 CLAS2로 교체하려면 독립 이진 head 또는 점수 구조 변경이 필요하며, 기본 검출 구조 변화까지 섞이므로 초기 비교에는 포함하지 않는다. MIL 기반 분류 손실 유지 여부는 이 차이를 설명한 뒤 사용자 판단 대상이며 현재 코드 수정은 하지 않았다.

여기서 정답은 영상의 감독 라벨과 PEL의 고정 의미 기준을 가리킨다. 라벨은 기존에도 학습되지 않았으며 현재 움직이는 것은 동적 프롬프트에서 나온 텍스트 특징이다. 추론에 정답 라벨을 입력하는 기능을 추가하지 않는다.

1. 전체 이벤트의 고정 문구를 동결된 CLIP의 `encode_text_original()`로 한 번 인코딩한다.
2. `no_grad`로 계산한 텍스트 기준을 정규화하고 캐시한다. 배치의 시각 특징이나 동적 Prompt Generator를 사용하지 않는다.
3. 최초 비교에서는 기존 label map의 텍스트를 고정 기준으로 사용하여 의미 문구 변경 효과를 섞지 않는다. 지식 기반 설명을 사용하려면 별도 입력 사전으로 제공하고 문구·추출기·버전·checksum을 기록한다.
4. 기존 이상 점수의 `detach()` 값으로 전경·배경 pooling 가중치를 만든다. 텍스트를 고정하는 것만으로는 선택 logits을 통한 Prompt Generator gradient가 차단되지 않는다.
5. 유효 길이만 사용하여 Temporal 출력 H를 pooling한다. H는 detach하지 않는다.
6. 모든 고유 클래스 텍스트를 denominator로 사용한다. 이상 전경은 해당 이벤트, 이상 배경과 정상 영상의 hard-negative pooling은 Normal을 목표로 한다.
7. 정규화된 pooled H와 고정 텍스트 기준의 cosine/temperature 및 클래스 목표 KL을 사용한다. 초기 계획은 기존 KL target의 logit smoothing 10을 보존하고, 후보 행렬만 M×M에서 M×C로 바꾼다. 클래스 인덱스는 canonical label map을 사용한다.

이 설계는 원본 PEL의 전경·배경 정렬을 활용한 **고정 전체 클래스 기준 변형**이다. 원본의 배치 내 반복 텍스트 후보와 같다고 표기하지 않는다. 새로운 projection, 이진 head, prototype 통신은 초기 비교에 추가하지 않는다.

| 경로 | 새 PEL 손실 | 기존 MIL 기반 분류 손실 |
| --- | --- | --- |
| Temporal과 위치 임베딩 | 업데이트 | 업데이트 |
| 동적 Prompt Generator | gradient 차단 | 기존대로 업데이트 |
| CLIP 파라미터 | 동결 | 동결 |
| 고정 PEL 텍스트 기준 | 값 고정 | 사용하지 않음 |
| 추론 | PEL 기준과 학습 라벨 불필요 | 기존 동적 프롬프트와 검출 경로 사용 |

따라서 의미 기준이 고정된다는 것과 모델의 모든 텍스트 관련 파라미터를 동결한다는 것은 다르다. 초기 비교는 PEL만 시각 경로에 작용하게 하고 기존 MIL의 프롬프트 학습은 유지한다. 검출에 쓰는 프롬프트까지 동결하거나 교체하는 것은 추가 구조 변경이므로 후속 비교 항목이다.

## 파일별 수정 계획

| 파일 | 계획 | 보존할 동작 |
| --- | --- | --- |
| `utils/config.py` | seed CLI와 명시적인 새 PEL variant 선택 추가. 기존 PEL mode는 기록 재현용으로 유지 | 기본값은 PEL off, 기존 결과 재현 가능 |
| `utils/pel_text_bank.py` 신규 | 클래스 순서, 고정 CLIP 텍스트 캐시, 선택적인 사전 입력과 식별 정보 처리 | optimizer parameter 및 집계 대상 아님 |
| `utils/tools.py` | 기존 loss를 보존하고 fixed-bank loss 추가. lengths, detached score, 전체 클래스 KL 및 인덱스 검증 | 전경·배경 가중식과 기본 weight/temperature의 의미 유지 |
| `utils/model.py` | 기존 보조 출력의 visual_features를 재사용. 필요한 경우 시각 출력만 요청하는 옵션 추가 | 기존 forward logits·state key·shape 유지 |
| `fed/client.py` | FedAvgClient에 선택적인 PEL 인자 전달과 MIL 기반 분류 손실+PEL 경로 추가. 고정 bank를 로컬 캐시 | optimizer, scheduler, loop, 글로벌 파라미터 반환과 배치 수 유지 |
| `fed/server.py` | FedAvgServer에서 PEL 인자를 FedAvgClient로 전달. 실행 설정과 실제 클래스 후보 로깅 | aggregate_parameters, send_global_parameter, 집계 계수 유지 |
| `train.py` | PEL flag를 기본 FedAvg에 직접 전달. seed 적용, 실행 manifest와 경로 기록 | FedStrategy/FedPGP로 우회하지 않음 |
| 평가·결과 저장 | round별 ROC/AP, final 및 checkpoint 정책을 기준/후보 동일하게 저장 | 첫 비교의 기존 검출 추론 유지 |
| 문서·검증 | 완료 시 코드 상태, 새 variant, gradient와 비교 결과 갱신 | 집계 트랙 결과와 과거 로그 보존 |

고정 텍스트 캐시는 학습 파라미터로 등록하지 않는다. inference checkpoint에 필요한 새 parameter key를 만들지 않고 설정·사전 식별값을 실험 manifest로 보존하는 방향을 우선한다.

## 유효 길이와 기준 모델 보존

새 PEL pooling은 padding 위치를 제외한다. 0 길이, 전부 padding, 라벨 누락과 0 batch는 명시적으로 오류 처리한다. 시각 특징을 정규화할 때 epsilon을 사용하고 유한 값을 확인한다.

최초 비교에서는 Temporal attention 및 Prompt Generator의 기존 padding 동작까지 함께 바꾸지 않는다. 이 부분을 수정하면 기준 모델도 같은 수정으로 다시 실행하고 별도 공통 변경으로 보고한다. PEL 개선에 기존 모델 수정 효과를 섞지 않는다.

## 비교 설계의 최소 범위

### 클라이언트 안에 여러 이벤트가 있는 조건

클라이언트가 여러 종류의 이벤트 영상을 소유하는 것은 기존 단일 라벨 UCF 데이터 형식과 호환된다. 각 영상의 class index를 사용하면 MIL 기반 분류 손실과 새 PEL 손실 모두 그대로 적용할 수 있다. 모든 클라이언트는 같은 14개 텍스트 기준과 클래스 순서를 공유하고, 자기 영상의 이벤트를 양성으로 선택한다. 이벤트 수나 빈도는 클라이언트마다 달라도 된다.

| 조건 | event 분할 | 여러 이벤트가 섞인 로컬 분할 |
| --- | --- | --- |
| 실제 시각 양성 | 로컬마다 이상 이벤트 하나 | 로컬마다 관측한 여러 이벤트 |
| 새 PEL 텍스트 후보 | 전체 14개 | 동일한 전체 14개 |
| 로컬 학습 효과 | 한 이벤트와 정상 분리, 다른 이벤트 텍스트와 구분 | 여러 이벤트를 각 의미 기준에 맞추고 로컬에서 이벤트 간 차이를 학습 |
| 남는 문제 | 이벤트 편향과 클라이언트 간 update 차이 | 빈도 편향, 희소 이벤트, 장면 차이와 update 충돌 |

다중 이벤트 로컬 데이터는 원본 PEL의 의미 구분을 활용하는 데 유리할 수 있다. 원본의 배치 기반 대조 후보는 실제 배치에 여러 이벤트가 있어야 늘어나지만 새 fixed-bank 손실은 배치에 한 이벤트만 있어도 전체 텍스트 후보를 사용한다. 따라서 새 손실을 위해 모든 batch를 다중 이벤트로 강제할 필요는 없다. 다중 이벤트에 의한 개선은 측정 전 가설이며 MIL 기반 분류 손실 기준 모델도 같은 혼합 데이터로 학습해야 한다.

1. 최초 event 비교를 유지한다. FedAvg와 MIL 기반 분류 손실 조건을 고정한 기준 및 후보를 같은 event 데이터로 비교한다.
2. 혼합 로컬 분할은 후속 별도 비교로 검토한다. 기존 UCF scene 분할은 활용 후보이지만 실제 라벨·카메라·세션 구성을 확인해야 하며 실제 지자체 데이터라고 단정하지 않는다.
3. 새 시뮬레이션 분할이 필요하면 같은 train pool의 각 영상 소유를 정하고, 클라이언트별 이벤트 종류·비율·데이터량을 기록한다. 균등 random은 서로 다른 지자체 분포를 충분히 모사하지 못할 수 있다. 생성 규칙·seed와 기존 test 보존을 명시한다.
4. 혼합 분할을 실제 생성하거나 기존 고객별 데이터를 옮기는 것은 이번 검토만으로 실행하지 않는다. 실행 범위가 지정되면 기존 분할과 별도 산출물로 준비한다.
5. 새 sampler나 class weight는 초기 비교에 넣지 않는다. 희소 이벤트를 놓치는지 로컬 클래스별 사용 횟수로 확인하고, 필요 시 별도 변경으로 기준과 후보 양쪽에 동일 적용한다.

다중 이벤트 분할 비교가 추가될 경우 아래 두 비교를 독립적으로 보고한다. event 후보와 mixed 기준을 직접 비교해 PEL 효과라고 해석하지 않는다.

| 분할 | 기준 | 후보 |
| --- | --- | --- |
| 기존 event | 기본 FedAvg + MIL 기반 분류 손실 | 동일 조건 + fixed-bank PEL |
| 혼합 로컬 | 기본 FedAvg + MIL 기반 분류 손실 | 동일 조건 + fixed-bank PEL |

한 클라이언트에 서로 다른 단일 라벨 영상이 여럿 있는 경우와 한 영상 자체에 여러 이벤트 라벨이 있는 경우는 다르다. 후자의 경우 `argmax` class index로 처리하면 일부 라벨을 버리므로 multi-positive 목표와 전경 분리 설계가 추가로 필요하다. 이번 UCF 단일 라벨 범위에는 해당 확장을 포함하지 않는다.

기존 PromptTranslator는 배치 시각 문맥을 평균하므로 혼합 이벤트 배치에 의해 검출 프롬프트도 변한다. 고정 PEL 기준은 이에 의존하지 않지만 MIL 기반 분류 손실 경로에는 남는다. 첫 비교에서는 기준과 후보에서 이 동작을 동일하게 유지하며 혼합 배치의 문맥 문제를 별도 구조 변경으로 해결하지 않는다.

| 조건 | 집계 | 데이터와 모델 | 손실 |
| --- | --- | --- | --- |
| 기준 | 기본 FedAvg | 전체 이벤트, 같은 분할과 초기화 | MIL 기반 분류 손실 |
| 후보 | 기본 FedAvg | 기준과 동일 | MIL 기반 분류 손실 + fixed-bank PEL |

PEL weight와 temperature의 초기값은 기존 구현 값을 출발점으로 기록하되, 성능 결과를 보기 전에 실행 조건으로 확정한다. 최초 비교에서 집계 및 문구 사전의 대규모 탐색은 하지 않는다. 모델 초기화, mini-batch 순서, 학습 step 수가 같도록 고정 bank 생성은 RNG를 소비하지 않게 구현한다.

실행 기록에는 seed, manifest, 실제 서버·클라이언트, 클래스 순서, PEL variant, 고정 bank checksum, temperature/weight, epoch/round, optimizer, LR, checkpoint 정책을 남긴다. 클래스별 로컬 샘플 수와 step 수는 진단 정보로 기록하며 집계 계수는 수정하지 않는다.

기존 코드의 best-on-test round는 논문/과거 로그와 대응하는 재현용 수치로 명시하고 final 점수와 전체 곡선을 함께 보고한다. 독립 성능 주장에는 validation에서 checkpoint를 선택하고 test를 최종 평가하는 조건을 기준과 후보에 똑같이 적용한다. validation 자료가 없는 현재 폴더에서 분할을 임의 생성하거나 실행 완료로 표기하지 않는다. 기존 논문 보고 수치와 새 로컬 기준 결과는 별도로 비교한다.

## 구현 순서와 검증

- [x] 기존 결과의 단일 로컬 및 연합 실행 단위 확인
- [x] 기본 FedAvg 고정과 PEL-only gradient 경계 설계
- [x] 전체 이벤트 텍스트 및 파일별 수정 계획 작성
- [x] 고정 bank 및 새 loss 구현 (`utils/fixed_pel.py`에 통합)
- [x] FedAvg 인자 전달 구현
- [x] 작은 텐서 및 옵션 off 회귀 검증
- [x] 사용자 Linux 환경에서 event/scene 기준 및 후보 실행 결과 보고받음 (독립 실행 확인과 구분)
- [x] 보고된 동일 조건 4개 실행의 최고·선택·최종 ROC/AP 비교 및 기록
- [x] 결과를 보고 후속 실험 설계: [호환성 진단·개선 계획](2026-10-08-pel-federated-compatibility.md)

| 검증 | 통과 조건 |
| --- | --- |
| bank 보존 | 학습 전후 checksum 동일, requires_grad=false |
| PEL-only backward | Temporal에 유한 gradient 존재. Prompt Generator 및 CLIP gradient 없음 |
| 전체 클래스 | UCF 후보 14개, CCTV 후보 7개. 로컬 배치에 없는 클래스도 실제 denominator에 존재 |
| padding | padding 위치의 값이 달라도 유효 H에 대한 pooling 결과 동일. Temporal 출력 변화 문제와 구분 |
| 옵션 off | 같은 seed와 입력에서 기존 logits, loss, 업데이트·집계 결과와 일치 |
| 집계 보존 | 반환 parameter key, 배포, 가중 단위와 local step 규약 동일 |
| 추론 | 학습 라벨 또는 PEL bank 없이 기존 checkpoint로 추론 가능 |
| 데이터 | class index, feature·GT 순서, stride와 예측 길이 일치 |

## 검증 기록과 남은 문제

구현 업데이트: [실행 및 산출물 안내](../../research/fixed-bank-pel-run.md). 기존 model/tools/inference/집계 수식은 유지했다. 2026-10-08에 [4개 실행 결과](../../research/2026-10-08-ucf-event-scene-results.md)를 기록했으며, 반복성·독립 검증과 [후속 진단](2026-10-08-pel-federated-compatibility.md)은 남아 있다. 아래 코드 미착수 기록은 변경 이전 분석이다.

2026-10-06 코드 정적 확인으로 legacy FedAvg의 PEL 전달 누락, standalone 단일 이벤트 dictionary와 인덱스 1 고정, 현재 PEL의 동적 text 및 logits gradient 경로를 확인했다. 계획 문서의 로컬 링크를 검사한다. 코드 구현, 학습과 성능 향상 확인은 아직 하지 않았다.

추가 원문 확인으로 MIL 기반 분류 손실이 Fed-WSVAD 기본 손실임을 확인하고, 원본 PEL4VAD의 CLAS2 이진 BCE 및 KL 정렬과 구분했다. 혼합 이벤트 로컬 구성은 적용 가능성 검토이며 기존 분할 변경이나 성능 개선 완료를 의미하지 않는다.

실제 데이터·GPU 환경·실행 예산 및 선택용 validation 제공 여부는 실행 단계에서 확인한다. 문헌 benchmark 보고치의 exact 설정과 새 실행 설정을 맞추기 전에는 SOTA 또는 재현 성공을 주장하지 않는다.
