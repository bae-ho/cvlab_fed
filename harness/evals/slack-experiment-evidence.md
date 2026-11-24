# Slack 실험 보고 근거 대조

- 사례 ID: slack-experiment-evidence
- 입력: [Slack PDF](../../docs/meeting-note/slack-experiments-log.pdf), [근거 대조](../../docs/research/slack-experiment-evidence.md), 원시 로그.
- 방식: 수동 정적 검토. 자동 실행기는 없다.

## 성공 기준
- 본문과 표 이미지 4개를 확인하고 날짜·물리 페이지를 남긴다.
- momentum 포함 UCF와 제외 UCF/XD 조합을 분리하고 Random UCF -0.11%p, Event XD -0.35%p를 보존한다.
- 61 미만 요약과 FedPGP 61.39 차이를 밝힌다.
- 최고 ROC round 10의 AP 64.89와 독립 최고 AP 65.29 round 1을 구분한다.
- ±4점 round 변동을 seed 분산으로 해석하지 않는다. seed/효율 계획을 완료로 표현하지 않는다.
- runs_4090 값 차이를 밝히고 run ID 없이 동일 실행으로 연결하지 않는다.
- 현행 목표와 두 트랙 경계를 유지하고 집계 보고를 PEL 작업의 집계 변경 지시로 바꾸지 않는다.

## 실패 조건
표 조건 상속, 이미지 누락, 없는 seed 결과 생성, 최고값만으로 SOTA 주장, 로컬 결과만으로 PEL 전체 폐기.

2026-10-06 PDF·로그 대조 내용을 근거 문서에 기록했다. 재평가 때는 별도 날짜와 판단을 남긴다.
