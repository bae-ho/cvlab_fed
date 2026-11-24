# UCF-Crime 이벤트별 클라이언트 분할의 PEL 의미 정렬 진단 결과

- 실험일: 2026-10-14; seed 88888888, FedAvg, 20 rounds, 기록 회차 run002
- [실험 기록](../../../docs/experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md)
- manifest.json: 설정·환경·revision·텍스트 기준 및 checksum
- round_metrics.jsonl: round 평가와 client별 손실·학습량
- pel_diagnostics.jsonl: client 업데이트 및 지정 batch gradient
- summary.json: 최고·최종 성능과 선택 정책
- analysis.json: 원시 SHA-256 및 모듈별 통계
- comparison.json: 기준 모델과의 round/module 대응 비교 및 gradient 표본 평균
- 원시 네 파일 내용 보존; 데이터·모델 가중치 제외
