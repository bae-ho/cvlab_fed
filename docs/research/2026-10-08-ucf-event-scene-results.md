# UCF event·scene 본 실험 결과

- 보고 정리일: 2026-10-08; 표의 날짜는 report-date
- 출처: 사용자가 제공한 네 실행의 전체 round 터미널 출력과 실행 명령; Linux 파일 자체·manifest·summary 미열람
- 상태: seed 88888888의 네 조건 완료 보고; 추가 seed 반복 미실행
- PEL 종료: event 2026-10-07 19:54:53 +09:00, scene 2026-10-08 16:09:24 +09:00; 각 종료 코드 0
- scene 기준 실행: 2026-10-08 13:26:01~14:45:52 +09:00, 종료 코드 0; event 기준 실제 실행일과 Linux commit·run ID 미확인

## 공통 조건

- UCF 전체 test의 프레임 ROC-AUC/AP; event 13 clients, scene 9 clients
- 기존 FedAvg, server optimizer none, personalization/local policy none
- MIL 기반 분류 손실 기준과 MIL 기반 분류 손실 + 1.0 × fixed-bank PEL 비교; temperature 0.09
- 기존 클래스 문구 14개 고정 기준; 지식 설명 JSON 미사용
- 20 round × 10 local epoch, batch size 64 per normal/anomaly loader, LR 0.0001, AdamW
- seed 88888888, 사용자 보고 RTX 3090 Ti·PyTorch 2.5.1+cu121
- 최고 checkpoint는 test ROC-AUC 최대 round; final은 round 20

## 결과 원시 정밀도

| 분할 | 방법 | 최고 ROC-AUC | 독립 최고 AP | 최고 ROC round의 AP | 마지막 ROC-AUC | 마지막 AP |
| --- | --- | ---: | --- | --- | ---: | ---: |
| event | 기준 | 0.8527366579931893 | 0.2865852068513203 | 0.26167553477541894 | 0.8523379086512399 | 0.2665676698745457 |
| event | PEL | 0.861960712247761 | 0.2968212240819206 | 0.28305119154995106 | 0.8399802397232 | 0.25301745355987004 |
| scene | 기준 | 0.8467936457438979 | 0.26974393292285803 | 0.26954786949140835 | 0.8383317264372787 | 0.24804702716894406 |
| scene | PEL | 0.8450755933287541 | 0.2729443800221195 | 0.2677944190675527 | 0.8231110160342242 | 0.24009097454501535 |

기준 대비 변화(반올림 전 계산, %p): event 최고 ROC +0.92, final ROC −1.24, final AP −1.36; scene 최고 ROC −0.17, final ROC −1.52, final AP −0.80.

추가 AP 변화: event 독립 최고 AP +1.02%p, 최고 ROC round의 AP +2.14%p; scene 독립 최고 AP +0.32%p, 최고 ROC round의 AP −0.18%p.

| 조건 | 최고 ROC round | 독립 최고 AP round |
| --- | ---: | ---: |
| event 기준 | 18 | 19 |
| event PEL | 11 | 12 |
| scene 기준 | 19 | 14 |
| scene PEL | 8 | 7 |

## AP 구분 및 보완 위치

- **Best AP**: round 전체 AP의 독립 최고값; `summary.json`의 `independent_best_ap`
- **Selected AP**: 최고 ROC checkpoint와 동일 round의 AP; `best_checkpoint_evaluation.ap`
- **Final AP**: 마지막 round의 AP; 현재 대화에서 전달된 값
- 네 실행의 각 20개 round 로그에서 최고 AP와 최고 ROC round의 AP를 계산해 보완 완료
- report provenance는 유지; 파일 자체와 checkpoint의 독립 검증은 미완료
- 초기 미보고 표는 보존하고 AP 보완 수정본을 별도 생성

## 소요 시간과 해석

| 분할·방법 | Wall time | 평균 round 시간 |
| --- | --- | --- |
| event 기준 | 미보고 | 미보고 |
| event PEL | 93분 53.312초 | 281.6656초 |
| scene 기준 | 79분 50.685초 | 239.53425초 |
| scene PEL | 79분 45.422초 | 239.2711초 |

평균 round 시간은 초기화·평가·저장까지 포함한 wall time/20. `user`·`sys` CPU 누적 시간과 구별. 전체 학습 속도 또는 시간 개선의 원인은 별도로 측정하지 않음.

- event: 최고 ROC 개선 관찰, 마지막 ROC/AP 하락 관찰
- scene: 보고된 최고 ROC와 마지막 ROC/AP 모두 하락 관찰
- 현재 결과로 모든 조건의 성능 향상 또는 혼합 이벤트 조건의 개선 주장 불가
- 독립 최고 AP는 두 분할 모두 증가; 최고 ROC checkpoint AP는 event만 증가
- 최고 AP round와 선택 checkpoint round가 모두 다르므로 최고 AP를 배포 checkpoint 성능으로 해석하지 않음
- round별 평가값 전부 확인; loss·업데이트 원인 및 추가 seed 안정성 검증 미완료
- 원본 PEL 전체 재현·SOTA·validation 선택 평가 완료 주장 없음

## 산출물과 검증

- [AP 보완 LaTeX 표](../../artifacts/experiment-tables/2026-10-08_ucf_event_scene_fedavg_fixed_bank_pel_seed88888888_round20_ap_completed.tex)
- [AP 보완 PNG 표](../../artifacts/experiment-tables/2026-10-08_ucf_event_scene_fedavg_fixed_bank_pel_seed88888888_round20_ap_completed.png)
- [네 실행의 회차 기록](../experiments/README.md), [JSON 기반 통합 비교표](../experiments/results-summary.md)
- [188개 측정값 CSV](../../artifacts/experiment-metrics/2026-10-08_ucf_event_scene_seed88888888_round20.csv)
- [보고 round 원시 정밀도 JSON](../../artifacts/experiment-metrics/2026-10-08_ucf_event_scene_reported_rounds.json)
- [기존 event 기록](2026-10-08-ucf-event-results.md)과 기존 표 보존
- 같은 LaTeX 원본을 300 DPI PNG로 렌더링; 원시 값·반올림·delta·분할 내 강조 대조
