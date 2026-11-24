# 0004: 집계 트랙 결합 실험 경로

- 날짜: 2026-11-11
- 상태: 채택
- 근거: 사용자 요청(집계 트랙 방식을 PEL 트랙에서 실험 가능하게)
- 관련: [두 트랙 경계](0003-two-track-pel-scope.md), [두 트랙 작업 범위](../research/two-track-scope.md)

0003이 고정해 둔 "PEL 작업 중 집계를 수정하지 않는다"는 경계는 유지한다. 이번 변경은 집계 **수식을 바꾸지 않고**, 집계 트랙이 이미 쓰는 설정을 PEL 트랙의 계측과 함께 실행할 수 있게 경로만 연다. `fed/aggregation.py`는 집계 트랙 저장소의 것과 바이트 단위로 동일하므로 새로 이식한 집계 코드는 없다.

## 무엇을 열었나

`--pel_allow_aggregation_integration`은 기존에 `fednova`/`fedpgp` 두 이름만 통과시키고 `--server_optimizer`를 무조건 막았다. 이제 이 플래그는 일반화 서버(`FedStrategyServer`)가 지원하는 세 축을 한 번에 연다.

- 집계 규칙: `fedavg` / `fedpgp` / `fedprox` / `scaffold` / `fednova`
- 서버 optimizer: `none` / `momentum` / `adam` / `yogi`
- 로컬 파라미터 정책: `none` / `adapter` / `norm` / `adapter+norm`

축 하나만 허용하지 않고 셋을 함께 연 이유는, PEL의 효과를 "어떤 집계 규칙 아래에서"로 읽으려면 같은 PEL 설정을 여러 규칙에 걸쳐 돌려야 하기 때문이다. 플래그 없이 기본 FedAvg로 실행하는 기존 경로는 그대로이고, 플래그를 켜고 기본 FedAvg를 그대로 두면 무의미한 조합이므로 거부한다.

`--use_fedpgp_contrastive`는 계속 막는다. 같은 공유 프롬프트 생성기에 보조 손실을 두 개 얹으면 gradient 진단이 어느 손실의 것인지 분해되지 않는데, 그 분해가 바로 진단의 목적이다.

## 계측을 같은 스키마로 맞춘 이유

`--pel_diagnostics`는 읽기 전용(옵티마이저에 쓰지 않고, 추가 forward·난수 소비 없음)이므로 집계 설정과 무관하게 동작할 수 있는데도 평면 FedAvg에만 연결돼 있었다. `FedStrategyServer`/`FedStrategyClient`에 동일한 측정을 연결하고, client 행의 키를 `FedAvgClient`와 일치시켰다. 결과적으로 `record_event_pel_comparison.py` · `summarize_pel_runs.py` · `analyze_pel_diagnostics.py`가 어느 서버가 만든 run인지 모르고도 읽는다.

`--pel_allow_aggregation_integration`이 켜진 run은 축이 legacy 서버와 등가로 풀리더라도 항상 `FedStrategyServer`를 쓴다. 라운드 단위 진단을 가진 유일한 서버이고, 거기서의 `fedpgp`는 `FedPGPServer`가 하는 공유 파라미터 평균과 수치적으로 같다.

## 진단 해석에서 주의할 점

`update_diagnostics`의 계수는 집계 규칙과 무관하게 **배치 수 가중치**로 고정했다. `retention_ratio`는 "이 클라이언트들이 서로 얼마나 상쇄되는가"라는 업데이트 자체의 성질이어야 규칙 간 비교가 되기 때문이다.

반면 `aggregate_rounding_residual_norm`은 평면 평균에서는 부동소수점 오차지만, FedNova나 서버 optimizer 아래에서는 **그 규칙이 가중 평균에서 벗어난 양**이고 크게 나오는 것이 정상이다. 이름만 보고 오독하지 않도록 레코드에 `aggregation_rule`을 함께 적고, FedNova의 실효 계수를 `rule_coefficients`(`tau_eff * p_i / tau_i`)로 병기한다.

## 남은 불일치 (이번 변경 범위 밖)

- 집계 가중치는 배치 수(`len(normal)+len(anomaly)`)인데 실제 스텝 수는 `min(두 로더)`이다. FedNova가 고치려는 불일치가 가중치 쪽에 남아 있다. 두 트랙 공통 문제이므로 별도로 결정한다.
- 두 트랙의 PEL 손실이 각각 구현돼 있고 정규화가 다르다(집계 트랙은 cross-entropy, 여기는 soft-target KL). 손실 값끼리 직접 비교하지 않는다.
- 집계 트랙의 UCF 프로토콜은 lr 1e-5 / 15 라운드, 이 트랙은 lr 1e-4 / 20 라운드다. 결합 run은 이 트랙의 프로토콜을 따르므로 집계 트랙 표의 수치와 직접 비교하지 않는다.
