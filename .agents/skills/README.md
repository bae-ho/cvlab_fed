# 프로젝트 스킬

| 스킬 | 사용 조건 |
| --- | --- |
| [reconcile-research-notes](reconcile-research-notes/SKILL.md) | 회의·제안서·Slack 근거 대조와 문서 갱신 |
| [validate-local-pel](validate-local-pel/SKILL.md) | 김배호 연구원 트랙의 PEL 적용점·실험 검증과 정체 분석 |
| [plan-federated-ablation](plan-federated-ablation/SKILL.md) | 요청한 트랙의 비교 설계. 현재 PEL 작업에서 집계 축은 고정 |

[두 트랙 경계](../../docs/research/two-track-scope.md)를 먼저 따른다. 스킬은 .agents/skills/<name>/SKILL.md에 둔다. 세션에서 검색되지 않으면 파일을 직접 읽는다. 학습을 자동 실행하는 실행기는 아니다.


- [export-cvpr-results](export-cvpr-results/SKILL.md): 실험 결과를 CVPR 스타일 LaTeX 원본과 날짜·실험명·조건을 담은 PNG/JPG로 함께 저장한다.


- [write-seminar-notes](write-seminar-notes/SKILL.md): 설계·실험에서 얻은 지식을 Mermaid·LaTeX·bullet point를 포함하는 세미나용 Markdown으로 기록한다.

- [record-experiment-results](record-experiment-results/SKILL.md): 날짜·실험종류·회차별 Markdown 누적 기록과 구조화된 측정값 CSV 추출.

record-experiment-results 호출 시 개별 회차 Markdown과 날짜순 통합표 results-summary.md를 함께 갱신한다.

모든 프로젝트 스킬의 문서 생성·정리·갱신은 [공통 용어 지침](../../docs/writing-guidelines.md)을 따른다. 공식 근거 없는 자체 축약 표현을 쓰지 않고 실험 대상·방법·조건을 이해할 수 있게 풀어서 적는다.
