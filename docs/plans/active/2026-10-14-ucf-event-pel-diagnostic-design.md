# UCF-Crime 이벤트별 클라이언트 분할에서 PEL 추가에 따른 손실 경쟁과 업데이트 상쇄 비교

- 작성일: 2026-10-14
- 상태: 실제 실행 결과 수신·분석 완료; [실험 기록](../../experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md). 사전 설계 내용은 아래에 보존
- 담당 트랙: 김배호 PEL 적용 지원
- 전체 계획: [연합학습 호환성 진단·개선](2026-10-08-pel-federated-compatibility.md)의 1단계
- 선행 결과: [PEL 미적용 기준 모델 진단 run002](../../experiments/2026-10-13_ucf_event_baseline_run002.md)
- 수식·지표 정의: [손실 경쟁과 업데이트 상쇄](../../seminar/2026-10-14_손실경쟁과_업데이트상쇄_진단.md)

## 왜 지금 이 실험을 하는가

PEL을 사용하지 않은 기준 모델에서도 client 업데이트의 부분 상쇄가 관측됐다. 예를 들어 round 5의 전체 공유 파라미터 보존 비율은 0.313946이고, 78개 client pair 중 30개가 음의 cosine을 보였다. 따라서 상쇄 자체를 PEL이 만들어 낸 문제로 설명할 수 없다.

이전 PEL 적용 실행은 최고 ROC가 개선됐지만 최종 성능은 기준보다 낮았다. 당시 로그에는 손실별 gradient와 client 업데이트 진단이 없어 원인을 판단할 수 없었다. 이번 목적은 PEL을 추가했을 때 기존 상쇄가 완화·악화되는지, 로컬 시각 모듈에서 두 손실의 경쟁이 나타나는지를 분리해서 확인하는 것이다. 성능 개선을 전제하지 않는다.

## 비교 설계

| 항목 | 완료된 기준 모델 | 이번 비교 모델 |
| --- | --- | --- |
| 데이터와 분할 | UCF-Crime, 이벤트별 13 clients | 동일 |
| 로컬 이상 이벤트 | client별 한 종류와 Normal | 동일 |
| 기본 검출 감독 | MIL 기반 분류 손실 | 동일 |
| 추가 의미 감독 | 없음 | 클래스 이름 14개 고정 텍스트 기준의 PEL 의미 정렬 손실 |
| 집계 | 기존 DataLoader 배치 수 기반 FedAvg | 동일 |
| 진단 | client 업데이트, 비활성 PEL 상태 | client 업데이트와 MIL/PEL gradient 경쟁 |

변경 요인은 PEL 손실 활성화 하나다. 현재 PEL은 원본 PEL4VAD 전체 재현이나 ConceptNet 사전 복원이 아닌 기존 fixed-bank 변형이다. 영상별 학습 라벨은 유지하며, 다른 client의 실제 영상 양성을 로컬로 옮기지 않는다.

## 고정 조건과 코드 경계

- seed 88888888; 20 rounds × 10 local epochs; 정상/이상 loader별 batch 64; AdamW, LR 0.0001
- server_optimizer none, personalization_type none, local_param_policy none
- 동일 feature·라벨·GT·분할 CSV·모델 초기화 설정·데이터 loader·추론의 1-P(Normal)
- PEL weight 1.0, temperature 0.09, score pooling, 전체 시각 gradient, 클래스 이름 bank
- ConceptNet, projection head, warmup, 클래스 기여도 보정, 집계 방식 변경 없음
- PEL은 기존 Temporal 출력에서 의미 정렬; 선택 점수와 고정 텍스트는 detach, 기본 분류 경로의 프롬프트 학습 유지
- 기준 manifest: commit cede515, PyTorch 2.5.1+cu121, git_dirty=true. 변경 파일 목록이 없어 소스 동일성 완전 확인 불가. 새 실행의 revision·dirty 및 데이터 변경 여부 기록; 기준 이후 변경은 문서·분석 도구 위주이나 두 작업 트리의 동일성을 추정으로 확정하지 않음

## 무엇을 측정하고 어떻게 판단하는가

| 질문 | 측정 | 비교·해석 기준 |
| --- | --- | --- |
| MIL과 PEL이 같은 시각 모듈에서 경쟁하는가? | 동일 forward의 gradient cosine, 각 norm, 계수 반영 norm 비율 | 음의 cosine이 여러 client·측정 round에서 반복되며 PEL norm도 무시할 수 없는지 확인; 손실 값 자체 비교로 충돌 판정 금지 |
| PEL이 client 업데이트 상쇄를 바꾸는가? | 모듈별 집계 보존 비율, 가중 pair cosine, 음의 pair 수, client/평균 업데이트 norm | 기준의 동일 round·모듈 대비 증가·감소를 함께 제시; 보존 비율만으로 성능 우열 판단 금지 |
| 후기 저하와 함께 나타나는가? | 전체 ROC/AP 곡선과 진단 round 1/5/10/15/20 | 후기 15/20의 변화와 성능 추이 대조; 최고값만 선택해 설명하지 않음 |
| 집계 측정이 실제 서버와 일치하는가? | 실제 집계 delta와 가중 delta의 잔차 | 반올림 잔차와 의미적 상쇄 구분; 큰 불일치 발견 시 원인 해석보다 구현 확인 우선 |
| 다른 학습량·클래스 노출의 영향인가? | client별 실제 steps·사용 클래스 횟수·평균 손실 | 기준과 대응 확인; local epoch 동일을 steps 동일로 오인하지 않음 |

업데이트는 매 round 측정. 손실 gradient는 각 지정 round의 client별 첫 epoch 첫 batch 표본으로 한정. 같은 seed라도 GPU 결정성·작업 트리·batch 동일성은 별도 확인하며 미확인인 경우 완벽히 짝지어진 batch 비교로 주장하지 않는다.

## 가능한 결과와 후속 판단

- PEL에서 방향 일치·보존 비율 증가: 공통 의미 감독의 정렬 가능성 지지; 성능도 개선되는지 별도 평가
- PEL에서 로컬 음의 gradient cosine과 상쇄 증가: 두 현상이 함께 나타나는 근거; 손실 강도·시점 후보의 우선 검토 근거
- 로컬 손실 경쟁만 증가하고 집계 변화가 작음: 로컬 경쟁과 집계 상쇄의 구분 유지; 집계를 원인으로 확정하지 않음
- 집계 정렬이 개선되지만 성능 하락: 정렬 자체가 유용한 이상 표현을 보장하지 않음; pooling·의미 기준·표현 변화 후속 검토
- 뚜렷한 변화 없음: 단일 첫-batch 표본과 단일 seed의 한계 기록; 가설 기각·확정을 과장하지 않음

임의의 절대 threshold로 충돌 여부를 이분화하지 않는다. 측정 round·client·모듈별 분포와 기준 대비 차이를 보고하며, 이후 손실 강도 대조 실험 전에는 성능 저하의 인과 원인으로 확정하지 않는다.

## 성능 평가와 완료 기준

- 최고 ROC·해당 round AP·독립 최고 AP·최종 ROC/AP·전체 곡선 모두 기록
- 기존 공통 test에서 최고 round 선택하는 탐색 정책 유지; 독립 validation으로 검증한 성능 주장과 구분
- 20 rounds, 13 clients, 지정 gradient 표본·유한 값·결측 사유·manifest·summary 대조 확인
- client 업데이트 수치와 같은 round의 평가를 연결; round를 독립 seed 표본으로 사용하지 않음
- 결과 수신 후 record-experiment-results로 새 실행 회차 및 통합표 갱신, write-seminar-notes로 관찰·가설·한계 갱신
- 단계 1 전체는 장면별 분할의 기준/PEL 진단도 마친 뒤 완료 판정; 이번 한 실행으로 종료하지 않음

## 실행 범위와 산출물

- 실행 대상: 사용자의 Linux RTX 3090 Ti 기존 환경, GPU 작업 여유 확인 후 한 조건만 실행
- 비용: 과거 같은 PEL 조건 약 94분; 진단 추가 비용 미측정으로 종료 시간 확정 불가
- 실행 명령: `bash scripts/run_pel_diagnostics.sh event_pel`; 여기서 마지막 인자는 스크립트 식별자이며 사용자 설명용 실험명과 구분
- 명령 안내 전 본 문서의 목적·조건·측정 설명 우선; 이 문서 작성으로 학습 자동 실행 없음
- 원시 보존: manifest.json, round_metrics.jsonl, pel_diagnostics.jsonl, summary.json 및 터미널 경고
- 정리 위치: artifacts/experiment-metrics 아래 실험 시작일·dataset·split·method·aggregation·seed·round·기록 회차를 담은 폴더. 실제 날짜와 실행 동일성을 확인한 뒤 이름 확정
- 기록 계열: 기존 ucf_event_fixed_bank_pel의 다음 회차; 회차는 학습 round와 구분

## 확인 기록

- [실행 스크립트](../../../scripts/run_pel_diagnostics.sh), [손실·업데이트 진단 코드](../../../utils/pel_diagnostics.py), [결과 기록 규약](../../../.agents/skills/record-experiment-results/SKILL.md)
- 이번 작업: 사전 설명·문서화 및 링크 확인만 수행; 학습·추가 CPU 합성 테스트 미실행
