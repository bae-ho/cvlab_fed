# 에이전트 작업 안내

## 시작 순서
1. README.md에서 프로젝트 실행 방법을 확인한다.
   연구 및 성능 작업은 docs/README.md의 현행 연구 목표와 성능 정의도 확인한다.
2. docs/architecture/README.md에서 코드 위치를 확인한다.
3. harness/checks/README.md에서 변경에 필요한 검증을 고른다.
4. 긴 작업은 harness/templates/task-plan.md를 복사해 docs/plans/active/에 기록한다.

## 작업 원칙
- 사용자 요청 범위 안에서 변경하고 기존 작업과 실험 결과를 보존한다.
- 실제 코드와 설정을 근거로 판단한다. 문서와 코드가 다르면 차이를 기록한다.
- 연구 목표와 성능 성공 기준의 원본은 docs/README.md이다. docs/meeting-note/의 과거 자료와 충돌하면 현행 정의 및 docs/decisions/의 변경 기록을 따른다.
- 데이터 경로, 비밀 값, 대형 데이터셋과 모델 가중치를 문서에 복사하지 않는다.
- 장시간 학습은 필요한 자원과 실행 범위가 확인된 경우에 수행한다.
- 새 실험은 실행 명령을 안내하거나 학습을 시작하기 전에 목적·설계 근거·비교군·변경 변수·통제 조건·판단 기준을 설명하고 사전 실험 문서에 저장한다. 사용자에게 실험을 설명할 때 스크립트 식별자만 쓰지 않고 데이터셋·클라이언트 분할·방법·검증 목적을 풀어서 적는다.
- 완료 보고에는 변경 내용, 검증 결과, 미검증 항목을 포함한다.
- 반복되는 절차는 .agents/skills/에, 설계와 근거는 docs/에 남긴다.

## 탐색 지도
- train.py / inference.py: 학습 및 추론 진입점
- fed/: 연합학습 코드
- utils/: 모델, 데이터셋, 설정 등
- scripts/: 실험 및 데이터 준비 스크립트
- AGGREGATION.md: 기존 집계 방식 문서
- harness/README.md: 하네스 운영 안내

## 연구 작업 안내
- 회의·제안서 작업은 docs/research/meeting-evidence.md와 implementation-status.md를 먼저 읽는다.
- 로컬 PEL 검증에는 .agents/skills/validate-local-pel/SKILL.md를, 어블레이션에는 .agents/skills/plan-federated-ablation/SKILL.md를 적용한다.
- 회의 기록 갱신에는 .agents/skills/reconcile-research-notes/SKILL.md를 적용한다.
- 연구 검증 기준은 harness/checks/research-validation.md, 기록 양식은 harness/templates/research-experiment.md에 있다.

## 두 트랙 작업 경계
- docs/research/two-track-scope.md를 먼저 읽는다. 장기태 연구원은 집계 개선, 김배호 연구원은 CCTV·PEL 적용을 독립 연구한다.
- 현재 사용자를 돕는 범위는 PEL 모듈 적용점과 실험 설계다. 집계 수식·가중치·서버 optimizer·공유/로컬 정책을 변경하지 않는다. PEL 옵션 미전달을 다른 집계 방식으로 우회하지 않는다.
- scene/event는 지자체 클라이언트를 염두에 둔 분할이다. PEL의 다중 이벤트 조건을 이유로 분할이나 데이터 소유 경계를 바꾸지 않는다. 두 분할의 호환성과 추후 통합 인터페이스를 기록한다.
- 가장 좋은 방법들의 통합은 후속 범위다. 현재 집계 트랙의 연구·실험·산출물을 보존한다.
- Slack 보고는 docs/research/slack-experiment-evidence.md에서 본문·표·로그 차이를 확인한다.


## 실험 결과표 산출물
- 결과표의 LaTeX·이미지 저장 요청에는 .agents/skills/export-cvpr-results/SKILL.md를 적용한다.
- 기본 저장 위치는 artifacts/experiment-tables/이며 실험날짜_실험이름및조건.tex와 같은 이름의 PNG/JPG를 함께 보존한다.


## 세미나용 지식 기록
- 지식 기록 요청이나 해당 스킬을 적용한 작업에서는 .agents/skills/write-seminar-notes/SKILL.md를 따른다.
- 기본 저장 위치는 docs/seminar/이며 기록일_주제.md로 저장한다. 도식은 Mermaid, 수식은 LaTeX, 중요 개념은 bullet point로 정리한다. 구현 사실·관찰·가설·제안을 구분하고 근거를 연결한다.

실험 결과 누적 기록과 수치 취합에는 .agents/skills/record-experiment-results/SKILL.md를 적용한다. 기본 위치는 docs/experiments/이며 날짜_실험종류_runNNN.md와 고정 JSON 측정값을 보존한다.

회차별 실험 결과를 기록·정정할 때 docs/experiments/results-summary.md도 원본 JSON에서 함께 갱신한다. 개별 파일과 통합 비교표의 수치·조건을 동기화한다.


## 모든 문서 작업의 용어 규칙
- docs/writing-guidelines.md를 반드시 따른다. 논문·명세·공식 문서에서 확인한 공식 용어 및 공식 약어만 사용하고, 자체적으로 만든 축약 표현은 쓰지 않는다.
- 공식 명칭이 없는 실험은 대상·방법·조건·목적이 이해되도록 풀어서 쓴다. 의미가 분명한 경우에만 조사를 생략하고 단어를 나열할 수 있다.
- 결과 기록·통합표·세미나 문서와 모든 문서 생성·정리·갱신 스킬, 제목·파일명·표·그림·완료 보고에 적용한다. 코드 식별자는 정확히 인용하고 완전한 설명을 병기한다. 새 스킬과 템플릿에도 이 규칙을 포함한다.
