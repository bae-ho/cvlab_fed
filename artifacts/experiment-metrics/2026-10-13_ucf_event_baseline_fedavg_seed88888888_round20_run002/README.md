# UCF event baseline 진단 원시 결과

- 실험일: 2026-10-13; 원 실행 ID `ucf-2026-10-13-17-48-25-320401`
- 저장 규칙: `YYYY-MM-DD_dataset_split_method_aggregation_seedN_roundN_runNNN/`
- 동일 계열 기존 run001 이후 새 실행 run002; 실제 round와 기록 회차 구분
- [공식 결과 기록](../../../docs/experiments/2026-10-13_ucf_event_baseline_run002.md)
- manifest.json: 실행 설정·코드 revision·환경
- round_metrics.jsonl: 20 rounds의 평가 및 client별 평균 손실·steps
- pel_diagnostics.jsonl: 20 rounds의 업데이트, 5개 지정 round의 첫 batch gradient
- summary.json: 최고·최종 평가
- analysis.json: 검증된 round/module 파생 통계 및 네 원시 파일 SHA-256
- 원시 네 파일 내용 수정 없음; 이전 시각 기반 폴더에서 의미 있는 조건 기반 이름으로 이동
- 재생성: `python scripts/analyze_pel_diagnostics.py <이 폴더> --output <새 JSON 경로>`
