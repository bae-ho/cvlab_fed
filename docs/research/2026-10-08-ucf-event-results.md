# UCF event 비교 결과

- 보고 정리일: 2026-10-08; 표 파일 날짜는 report-date
- 실행일: PEL 종료 2026-10-07 19:54:53 +09:00; 기준의 실제 실행일 미확인
- 출처: 사용자가 대화에 제공한 터미널 출력 및 앞서 실행한 명령
- 상태: 두 본 학습 완료 보고; 원시 로그·manifest·checkpoint의 독립 확인 미완료
- 실행 환경: 사용자 보고 RTX 3090 Ti, Python 3.11 환경, PyTorch 2.5.1+cu121

## 통제 조건

- UCF event, 13 clients, 공통 전체 test 평가
- 기본 FedAvg, 서버 optimizer none, 개인화·로컬 parameter policy none
- 20 round × 10 local epoch, seed 88888888, learning rate 0.0001, AdamW
- 정상·이상 loader 각각 batch size 64; 결합 batch 최대 128
- 기준 MIL 기반 분류 손실 / 후보 MIL 기반 분류 손실 + 1.0 × fixed-bank PEL, temperature 0.09
- 고정 텍스트 후보 14개; 기본 클래스 문구 사용, 추가 지식 설명 JSON 미사용
- 현재 코드의 집계 가중 단위는 정상·이상 loader 배치 수 합; 변경 없음
- 최고 test ROC-AUC round 선택; final은 round 20
- Linux 실제 commit·run 폴더·데이터 manifest·완전 결정성 미확인

## 결과

| 조건 | 최고 ROC-AUC (0–1) | 최고 ROC round의 AP | 마지막 ROC-AUC (0–1) | 마지막 AP (0–1) |
| --- | ---: | --- | ---: | ---: |
| 기준 | 0.8527366579931893 | 0.26167553477541894 | 0.8523379086512399 | 0.2665676698745457 |
| fixed-bank PEL | 0.861960712247761 | 0.28305119154995106 | 0.8399802397232 | 0.25301745355987004 |

- 기준 대비 변화: 최고 ROC +0.9224%p, 마지막 ROC −1.2358%p, 마지막 AP −1.3550%p
- PEL wall time 93분 53.312초; 종료 코드 0
- 평균 round 소요 281.6656초; 초기화·평가·저장 포함 전체 wall time을 20으로 나눈 값, 순수 로컬 학습 속도와 구별
- 전체 로그 보완: 최고 ROC는 기준 round 18, PEL round 11; 독립 최고 AP는 기준 0.2865852068513203 (round 19), PEL 0.2968212240819206 (round 12)
- 기준 소요 시간만 미보고; AP 보완값은 [네 조건의 수정 기록](2026-10-08-ucf-event-scene-results.md)과 구조화된 회차 JSON에도 반영
- `user`/`sys`는 프로세스·스레드 CPU 시간 합; wall time 또는 GPU 계산 시간으로 해석하지 않음

## 해석과 다음 확인

- 단일 seed에서 test 기준 최고 ROC 향상 관찰
- 마지막 round ROC/AP 하락 관찰; 모든 지표 또는 학습 안정성의 개선으로 확대 해석하지 않음
- 후기 과적합·업데이트 충돌 등 원인은 가설; round별 평가값은 확보했지만 loss·업데이트 근거 없이 확정 불가
- scene 두 조건도 완료; [통합 결과와 AP 보완 표](2026-10-08-ucf-event-scene-results.md)에서 비교
- 추가 seed 및 validation 선택 평가 미완료; SOTA 달성 주장 없음

## 표 산출물과 근거

- [LaTeX 원본](../../artifacts/experiment-tables/2026-10-08_ucf_event_fedavg_fixed_bank_pel_seed88888888_round20.tex)
- [PNG 결과표](../../artifacts/experiment-tables/2026-10-08_ucf_event_fedavg_fixed_bank_pel_seed88888888_round20.png)
- 위 초기 event 표는 AP 미보고 당시의 기록으로 보존; 현행 AP 보완 표는 [통합 결과 문서](2026-10-08-ucf-event-scene-results.md) 참조
- [실행 안내](fixed-bank-pel-run.md), [현행 성능 목표](../README.md#현행-연구-목표와-성능-정의)

## 산출물 확인

- 같은 LaTeX 원본으로 PNG 300 DPI 생성; Times·booktabs·상단 caption 적용
- 원시 정밀도와 보고 출처를 원본 주석에 보존; 표시값과 delta 재계산 대조
- PNG의 글자·강조·수치·잘림 육안 확인
- renderer의 양 열 폭 사용 시 바깥 문서 폭 불일치 수정; 실제 표 overflow 검사는 유지
