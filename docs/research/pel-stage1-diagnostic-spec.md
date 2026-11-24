# 1단계 — MIL 기반 분류 손실·PEL 손실 및 연합 업데이트 진단 명세

- 명세일: 2026-10-13
- 상태: 이벤트별 클라이언트 분할의 기준/PEL 진단 완료; 장면별 클라이언트 분할의 두 진단 미완료
- 2026-10-14 갱신: 다음 실험은 [PEL 추가에 따른 손실 경쟁·업데이트 상쇄 사전 설계](../plans/active/2026-10-14-ucf-event-pel-diagnostic-design.md)를 먼저 설명한 뒤 실행
- 계획 원본: [호환성 진단·개선 계획](../plans/active/2026-10-08-pel-federated-compatibility.md)
- 목적: 후기 성능 저하와 공유 시각 모듈의 손실 경쟁·클라이언트 업데이트 상쇄 사이의 관계 확인

## 실행 조건

| 순서 | 실험 설명 및 스크립트 식별자 | 분할 | clients | 손실 |
| --- | --- | --- | ---: | --- |
| 1 | UCF-Crime 이벤트별 클라이언트의 PEL 미적용 기준 진단 (`event_baseline`) | event | 13 | MIL 기반 분류 손실 |
| 2 | UCF-Crime 이벤트별 클라이언트의 PEL 추가 손실 경쟁·상쇄 진단 (`event_pel`) | event | 13 | MIL 기반 분류 손실 + fixed-bank PEL |
| 3 | UCF-Crime 장면별 클라이언트의 PEL 미적용 기준 진단 (`scene_baseline`) | scene | 9 | MIL 기반 분류 손실 |
| 4 | UCF-Crime 장면별 클라이언트의 PEL 추가 손실 경쟁·상쇄 진단 (`scene_pel`) | scene | 9 | MIL 기반 분류 손실 + fixed-bank PEL |

공통 seed 88888888, 20 rounds, 10 local epochs, 정상/이상 loader별 batch 64, learning rate 0.0001. 기존 AdamW와 DataLoader 배치 수 기반 FedAvg 유지. 서버 optimizer·개인화·로컬 정책 none. PEL은 weight 1.0, temperature 0.09, 클래스 이름 14개 bank, score pooling, gradient scope all. ConceptNet·새 head·warmup·클래스 보정·집계 변경은 이번 비교에서 사용하지 않음.

기존 4개 실행은 ROC/AP 근거로 보존. 손실·gradient 로그가 없어 새 진단 실행을 대체하지 못함. 기준 실행의 PEL gradient는 비활성에 따른 결측이며 손실 충돌 없음의 증거가 아님.

## 측정과 해석

- 매 round: client별 평균 MIL 기반 분류 손실/PEL/total, 실제 steps, 사용 클래스 수, ROC/AP
- rounds 1/5/10/15/20: 각 client 첫 epoch 첫 batch의 동일 forward에서 두 손실 gradient norm·cosine, PEL 계수 적용 전후 norm 비율
- 같은 진단 batch: 전경/배경 pooling 집중도와 텍스트 유사도, 클래스별 PEL 기여
- 매 round: 공유 모듈 client delta norm·pairwise cosine·집계 보존 비율·실제 집계 잔차
- 손실 경쟁의 근거: 공유 시각 모듈의 음의 cosine과 유의한 PEL norm이 반복되고 후기 저하 시점과 함께 나타나는지 확인
- 업데이트 상쇄의 근거: 기준 대비 PEL에서 방향 불일치·낮은 집계 보존 비율이 지속되는지 비교
- 제한: 손실 값의 동반 상승만으로 gradient 충돌 확정 불가. FedAvg 자체에 별도의 미분 가능한 손실 없음. 첫 batch 표본은 전체 gradient 분포를 대표하지 않을 수 있음. 단일 seed의 상관관계는 인과 증명 아님
- 성능 보고: 최고 ROC·그 round AP·독립 최고 AP·최종 ROC/AP를 분리. 기존 test 기반 최고 선택은 탐색용

## Linux 실행

기존 학습 환경에서 프로젝트 루트로 이동하고 최신 main을 받은 뒤 아래를 하나씩 실행. 다른 GPU 작업이 끝난 뒤 시작. 호스트 드라이버·CUDA·기존 Docker 환경 변경 불필요.

```bash
git pull --ff-only origin main
bash scripts/run_pel_diagnostics.sh event_baseline
```

아래는 후속 조건의 명령 목록이며 연속 자동 실행 지시가 아니다. 각 새 실험의 목적·설계·비교·판단 기준 설명과 사전 문서화를 마친 뒤 해당 한 조건만 실행한다. 완료된 첫 기준 진단을 재실행할 필요는 없다.

```bash
bash scripts/run_pel_diagnostics.sh event_pel
bash scripts/run_pel_diagnostics.sh scene_baseline
bash scripts/run_pel_diagnostics.sh scene_pel
```

스크립트 한 번은 지정한 한 조건만 실행. 실패 시 중단하며 타임스탬프별 터미널 로그 보존. 저장 위치는 기존 `save/<dataset-timestamp>/` 정책 유지. 터미널 로그 외에 각 실행의 `manifest.json`, `round_metrics.jsonl`, `pel_diagnostics.jsonl`, `summary.json`을 함께 보존. 기존 checkpoint 및 데이터 경로 수정 없음.

기존 비용은 event PEL 약 94분, scene 기준/PEL 각각 약 80분. 진단의 추가 CPU RAM·동기화·시간 비용은 미측정이므로 전체 완료 시간 확정 불가.

## 완료 기준과 기록

네 조건의 20 rounds 및 진단 파일 확인 후 손실·gradient·업데이트·평가 추이 대조. 결측·중단은 별도 상태로 기록. 결과는 record-experiment-results의 새 실행 회차와 통합표에 저장하고, 관찰·가설·판단은 write-seminar-notes로 기존 5단계 노트에 갱신. 로그가 없는 현재 상태를 완료 결과로 기록하지 않음.
