# 회차별 실험 결과

[record-experiment-results 스킬](../../.agents/skills/record-experiment-results/SKILL.md)로 실험 결과를 계속 기록한다. 파일명은 YYYY-MM-DD_실험종류_runNNN.md이며 동일 종류는 날짜가 달라도 회차를 이어간다. 회차는 학습 round와 다르다.

문서에는 해석·근거와 고정 experiment-results JSON 블록을 함께 보존한다. 숫자는 numeric, 미측정은 null이며 단위·조건·seed·선택 기준을 분리한다. 수치 추출은 스킬의 extract_results.py로 CSV를 만든다. 생성된 CSV는 원본 문서에서 재생성 가능한 파생 산출물이다.

## 기록 목록

사용자 제공 전체 round 로그에서 네 실행을 각각 기존 실행의 run001로 최초 등록했다. 재학습 결과가 아니며 상태는 reported다. Event 기준의 실행일만 미확인이므로 보고일을 사용했다.

| 날짜 | 실험 종류 | 회차 | 상태 | 개별 기록 |
| --- | --- | ---: | --- | --- |
| 2026-11-12 (실험일) | ucf_fednova_adapter_pel_stagewise | 1 | completed | [집계 규칙과 의미 정렬 손실의 단계별 적용 효과](2026-11-12_ucf_fednova_adapter_pel_stagewise_run001.md) |
| 2026-10-14 (실험일) | ucf_event_fixed_bank_pel | 2 | completed | [UCF-Crime 이벤트별 클라이언트의 PEL 의미 정렬 진단](2026-10-14_ucf_event_fixed_bank_pel_run002.md) |
| 2026-10-13 (실험일) | ucf_event_baseline | 2 | completed | [event 기준 진단](2026-10-13_ucf_event_baseline_run002.md) |
| 2026-10-07 (실험일) | ucf_event_fixed_bank_pel | 1 | reported | [event PEL](2026-10-07_ucf_event_fixed_bank_pel_run001.md) |
| 2026-10-08 (보고일) | ucf_event_baseline | 1 | reported | [event 기준](2026-10-08_ucf_event_baseline_run001.md) |
| 2026-10-08 (실험일) | ucf_scene_baseline | 1 | reported | [scene 기준](2026-10-08_ucf_scene_baseline_run001.md) |
| 2026-10-08 (실험일) | ucf_scene_fixed_bank_pel | 1 | reported | [scene PEL](2026-10-08_ucf_scene_fixed_bank_pel_run001.md) |


## 전체 수치 비교
[날짜순 통합 비교표](results-summary.md)는 개별 회차 JSON에서 자동 생성한다. 결과 기록·정정 때마다 함께 갱신하며 수치는 metric/단위별 열로 비교한다. 통합표는 직접 편집하지 않는다. 기존 네 실행과 2026-10-13 기준 진단 실행을 포함한다.

개별 결과 문서와 통합표 모두 [공통 용어 지침](../writing-guidelines.md)을 따른다. 임의 축약 대신 공식 용어 또는 실험 대상·방법·조건을 설명하는 표현을 사용한다. 기계 판독용 식별자는 설명과 대응시키며 기존 기록과의 연결을 보존한다.
