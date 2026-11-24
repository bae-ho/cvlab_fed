# Slack 실험 결과와 근거 대조

정리일: 2026-10-06. 원본 S: [slack-experiments-log.pdf](../meeting-note/slack-experiments-log.pdf), 3쪽. 본문과 결과표 이미지 4개를 확인했다. 기존 보고·로그 요약이며 새 실험이 아니다. [현행 목표](../README.md#현행-연구-목표와-성능-정의)와 [두 트랙 경계](two-track-scope.md)를 우선한다.

| ID | 날짜 / 페이지 | 보고 | 해석 |
| --- | --- | --- | --- |
| S01 | 8/31 / p.1 | FedPGP event 하락, scene 상승 | 결과표 존재. Non-IID 원인 설명은 가설 |
| S02 | 8/31 / p.1 | 다른 집계도 비교하자는 피드백 | 제안이며 최종 채택 아님 |
| S03 | 9/11 / pp.2–3 | FedNova+momentum+low-rank adapter UCF 전 분할 개선 | seed 검증 진행 중. 효율 및 XD 검증은 본문에서 계획 |
| S04 | 날짜 별도 없음 / p.2 | momentum 제외 FedNova+adapter UCF/XD 통합표 | S03과 다른 조합. XD 수치 존재, 날짜·seed·run ID 미제공 |
| S05 | 10/2 / p.3 | CCTV 정체, PEL 후기 ROC 약 50, Fight 외 로컬 개선 없음 | 표·로그와 요약 일부 불일치. 원인·유의성 미확정 |

## S01: FedPGP 표
ROC/AP 0–1 원표를 %로 변환했다. delta는 %p다. 본문이 데이터셋을 명시하지 않아 UCF로 확정하지 않는다.

| split / clients | FedAvg ROC / AP (%) | FedPGP global ROC / AP (%) | delta ROC / AP (%p) |
| --- | --- | --- | --- |
| Scene / 9 | 83.99 / 23.35 | 84.74 / 26.52 | +0.75 / +3.17 |
| Event / 13 | 85.90 / 28.57 | 85.30 / 26.26 | -0.60 / -2.31 |

[현재 event 로그](../../runs_4090/ucf_event_fedpgp.log) 최고 ROC 84.97%, [scene 로그](../../runs_4090/ucf_scene_fedpgp.log) 84.70%는 표와 일치하지 않는다. 동일 실행으로 연결하지 않는다. Scene의 Non-IID가 더 크다는 해석은 클래스·도메인 분포로 별도 검증한다.

## S03: momentum 포함 UCF 조합
| split | baseline (%) | 조합 (%) | delta (%p) |
| --- | --- | --- | --- |
| Random | 83.78 | 84.89 | +1.11 |
| Event | 84.83 | 85.40 | +0.57 |
| Scene | 83.99 | 84.37 | +0.38 |

표는 지표명을 생략했다. UCF AUC로 사용할 때 원 실행 정의를 확인한다. seed 초기화 검증 진행 보고를 다중 seed 안정성 완료로 바꾸지 않는다.

## S04: momentum 제외 통합표
표 설정: UCF visual_layers=2, XD visual_layers=1; lr=1e-5, 15 rounds, 10 local epochs. UCF AUC%, XD AP%. 논문 보고치 (Ours)는 원논문과 비교 설정 확인 전 현재 달성 수치로 사용하지 않는다.

| 조건 | Random UCF | Random XD | Event UCF | Event XD | Scene UCF | Scene XD |
| --- | --- | --- | --- | --- | --- | --- |
| 논문 보고치 (원표 Ours) | 84.06 | 75.32 | 85.07 | 74.23 | 84.03 | 75.99 |
| baseline FedAvg | 83.78 | 74.50 | 84.83 | 77.10 | 83.99 | 75.80 |
| FedNova + adapter, momentum 제외 | 83.67 | 74.74 | 85.04 | 76.75 | 85.05 | 77.61 |
| delta (%p) | -0.11 | +0.24 | +0.21 | -0.35 | +1.06 | +1.81 |

Random UCF와 Event XD는 하락한다. S03의 전 분할 개선을 확대하지 않는다. 표에 없는 low-rank 여부·seed·batch size를 다른 표에서 상속하지 않는다.

## S05: CCTV 표와 원시 로그
원표 Fed + PEL4VAD 수치는 FedPGP+PEL 로그와 반올림 수준에서 일치한다. 명칭만으로 옵션 활성화를 증명하지 않는다.

| 방법 | 최고 ROC (%) | 해당 round AP (%) | ROC 최고 round | round 20 ROC (%) | 독립 최고 AP (%) / round |
| --- | --- | --- | --- | --- | --- |
| FedAvg | 60.22 | 65.01 | 1 | 52.30 | 65.01 / 1 |
| FedPGP | 61.39 | 64.89 | 10 | 54.06 | 65.29 / 1 |
| FedPGP + PEL | 59.51 | 63.99 | 1 | 50.32 | 63.99 / 1 |

로그: [FedAvg](../../logs/cctv_fedavg_train.log), [FedPGP](../../logs/cctv_fedpgp_train.log), [FedPGP+PEL](../../logs/cctv_fedpgp_pel4vad_train.log).

- 모두 ROC 61 미만이라는 본문은 FedPGP 61.39와 충돌한다. 대부분 첫 round 최고는 FedAvg/PEL에 해당하며 FedPGP는 round 10이다.
- 원표 Best AP 64.89는 최고 ROC round의 AP다. 독립 최고 AP는 65.29다.
- 최고 ROC 차이는 +1.17%p. 본문의 ±4점은 round 변동 서술이며 seed 표준편차·신뢰구간이 아니다. 통계적 유의/무의를 확정하지 않는다.
- PEL round 8 ROC 49.99%, round 20 50.32%로 정체가 보이나 원인은 미확정이다.
- 기존 다중 seed 평균도 Fight 상승·다른 CCTV 이벤트 하락으로 요약된다. 개별 seed 전부 하락 또는 PEL 전체 기각으로 확대하지 않는다. [PEL 분석](../architecture/pel4vad-integration.md)

## 후속 확인
집계 트랙의 seed·비용 검증과 실행 로그 연결은 아직 미확인이다. 현재 PEL 작업은 집계를 유지하며 CCTV의 전체 round 곡선, gradient·update, 유효 길이, 실제 대조 후보와 의미 기준을 확인한다. 효율 실측 및 두 트랙 통합은 별도 후속 범위다. 최종 채택·SOTA 달성을 이 보고로 확정하지 않는다.
