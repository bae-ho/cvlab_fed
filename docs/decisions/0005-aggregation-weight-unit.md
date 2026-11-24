# 0005: 집계 가중치 단위

- 날짜: 2026-11-11
- 상태: 채택 (기본값은 기존 단위 유지)
- 근거: 코드 검토에서 발견한 가중치·스텝 수 불일치
- 관련: [집계 트랙 결합](0004-aggregation-track-integration.md)

## 문제

ucf/cctv 클라이언트는 normal 로더와 anomaly 로더를 **한 배치씩 짝지어** 학습하므로 epoch당 스텝 수는 `min(len(normal), len(anomaly))`이다. 그런데 서버에 보고하는 가중치 n_i는 `len(normal) + len(anomaly)`였다. 두 로더 길이가 다른 클라이언트는 실제로 수행한 것보다 큰 가중치를 받는다.

UCF scene 분할 실측 (batch 64):

| client | (normal, anomaly) | 기존 n_i | 실제 스텝/epoch | 정규화 기존 → 수정 |
|---|---|---|---|---|
| 0 | (41, 56) | 97 | 41 | 0.3992 → 0.4141 |
| 1 | (32, 28) | 60 | 28 | 0.2469 → 0.2828 |
| **2** | **(17, 4)** | **21** | **4** | **0.0864 → 0.0404 (0.47배)** |
| 7 | (3, 1) | 4 | 1 | 0.0165 → 0.0101 |

event 분할은 두 로더가 거의 균형이라 최대 괴리가 0.93배, random은 1.03배로 사실상 영향이 없다. **scene에서만 유의미하게 틀어진다.**

이것이 중요한 이유는 FedNova가 겨냥하는 대상이 바로 스텝 수 편차이기 때문이다. τ_i로 나눠 정규화해 놓고 가중치 p_i는 다른 단위를 쓰면, 고치려던 불일치가 가중치 쪽에 남는다.

## 결정

`--aggregation_weight_unit`을 추가한다.

- `loader_batches` (**기본값**): `len(normal) + len(anomaly)`. 기존 동작 그대로.
- `effective_batches`: `min(len(normal), len(anomaly))`. 실제로 소비되는 배치 수.

xd는 클라이언트당 짝 없는 로더 하나라 두 단위가 같은 값을 돌려준다.

기본값을 바꾸지 않은 이유는 `docs/experiments/`의 기존 기록이 모두 `loader_batches`로 생성됐고, 기본값을 바꾸면 새 run이 그것들과 조용히 비교 불가능해지기 때문이다. 선택한 단위는 `manifest.json`과 `pel_diagnostics.jsonl`의 `aggregation_unit`에 기록되며, `summarize_pel_runs.py`의 조건 키에 자동으로 들어가므로 단위가 다른 run은 짝지어지지 않는다.

## FedNova와의 상호작용

`effective_batches`에서는 p_i ∝ τ_i / local_epochs가 되어 두 양이 비례한다. 이때 FedNova의 delta는

    delta = (sum tau^2 / (sum tau)^2) * sum_i (w_i - w)

로 정리된다 — 정규화 평균이 클라이언트 업데이트의 **단순 합**에 스칼라를 곱한 형태로 환원된다. 이는 p_i ∝ τ_i일 때의 FedNova 고전적 결과이고 버그가 아니다. 다만 FedNova와 FedAvg의 차이가 이 단위에서는 줄어들 수 있으므로, 집계 규칙 비교는 **단위를 고정한 채** 수행한다.

## 고치지 않은 것

scene client 2는 normal 17배치 중 4배치만 epoch마다 소비한다(shuffle이 있어 epoch마다 다른 4배치). 즉 **보유 정상 데이터의 상당 부분이 라운드마다 학습에 쓰이지 않는다.** 이는 가중치 단위가 아니라 학습 루프의 페어링 방식 문제이고, 고치려면 normal 로더를 순환시키는 등 루프 구조를 바꿔야 하므로 이번 범위에서 제외한다. 별도 결정이 필요하다.
