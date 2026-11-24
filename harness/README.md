# 하네스 운영

하네스는 에이전트가 작업 맥락을 읽고, 작업을 수행하고, 결과를 검증하며, 재사용 가능한 지식을 남기는 구조다. 이 디렉터리는 운영 규약을 담는다. 자동 실행기와 CI는 아직 연결하지 않았다.

## 구조
- ../AGENTS.md: 에이전트 진입점과 작업 규칙
- ../.agents/skills/: 재사용 가능한 스킬
- ../docs/architecture/: 코드 구조와 설계
- ../docs/plans/active/: 진행 중인 작업 계획
- ../docs/plans/completed/: 완료된 작업 기록
- ../docs/decisions/: 설계 결정과 이유
- checks/: 변경별 검증 기준
- evals/: 에이전트 작업 품질 평가 사례
- templates/: 계획과 평가 사례 템플릿

## 작업 흐름
1. 요청과 성공 기준을 확인한다.
2. 관련 코드와 문서를 읽고 필요한 경우 계획을 기록한다.
3. 변경한 뒤 해당 범위의 검증을 수행한다.
4. 결과와 실행 환경을 기록하고 계획을 completed로 옮긴다.
5. 반복 절차는 스킬로 정리하고 발견한 실패는 평가 사례로 추가한다.

실험 로그는 기존 logs/와 runs_4090/를 사용한다. 문서에는 로그 위치와 필요한 요약만 남긴다.

## 연구 작업에 사용하기
- [프로젝트 스킬](../.agents/skills/README.md)
- [회의 근거](../docs/research/meeting-evidence.md)와 [현재 구현 차이](../docs/research/implementation-status.md)
- [연구 검증 기준](checks/research-validation.md)
- [실험 기록 템플릿](templates/research-experiment.md)
- [수동 회귀 평가 사례](evals/research-regressions.md)

## 현재 지원 범위
[두 트랙 작업 경계](../docs/research/two-track-scope.md)에 따라 김배호 연구원의 PEL 적용점과 실험 설계를 돕는다. 집계 축은 고정하며 scene/event 호환성과 추후 통합 인터페이스를 확인한다. [Slack 근거](../docs/research/slack-experiment-evidence.md)도 함께 읽는다.


## 실험 결과표 내보내기
[CVPR 결과표 스킬](../.agents/skills/export-cvpr-results/SKILL.md)은 raw LaTeX 표 원본을 저장하고 실제 컴파일 결과를 PNG/JPG로 만든다. 기본 경로는 artifacts/experiment-tables/이며 실험 날짜와 조건을 파일명에 담는다.


## 세미나용 지식 축적
[지식 기록 스킬](../.agents/skills/write-seminar-notes/SKILL.md)로 확인한 구조·계산·실험 흐름을 [세미나 노트](../docs/seminar/README.md)에 저장한다. Mermaid·LaTeX 원본을 보존하고 중요 개념과 근거를 함께 정리한다.

- [회차별 실험 기록 스킬](../.agents/skills/record-experiment-results/SKILL.md)은 docs/experiments/에 날짜·종류·회차별 Markdown을 보존하고 고정 JSON에서 수치를 CSV로 추출한다.

문서 작성·정리·갱신 전반에 [공통 용어 지침](../docs/writing-guidelines.md)을 적용한다. 자체 축약 표현을 금지하고 공식 근거와 실험 설명의 명확성을 확인한다.
