---
name: record-experiment-results
description: 실험 결과를 날짜·실험종류·회차별 Markdown 문서에 누적 기록하고 고정 JSON 측정값을 보존하여 날짜순 통합 Markdown 비교표를 매번 함께 갱신하고 여러 문서의 수치를 CSV로 추출한다. 결과 기록·회차 관리·수치 취합 요청에 사용한다.
---

# 회차별 실험 결과 기록

## 기록 범위
관련 로그·보고표와 실행 설정을 읽고 확인된 결과만 기록한다. [두 트랙 경계](../../../docs/research/two-track-scope.md)를 따른다. 기록을 위해 학습하거나 집계 설정을 변경하지 않는다. 과거 보고, 실제 실행, 재현 완료는 구분한다.

## 파일명과 누적 규칙
사용자 지정 경로를 우선하며 기본은 `docs/experiments/YYYY-MM-DD_실험종류_runNNN.md`이다. 첫 실행도 run001로 둔다. 예: `2026-10-08_cctv_event_pel_weight_run001.md`.

날짜는 실제 실험일을 사용한다. 보고일만 있으면 date_kind=report로 표시한다. 둘 다 모르면 확인하고 현재 날짜를 실제 실험일로 추정하지 않는다. 파일명 date와 JSON date를 일치시킨다.

experiment_type은 반복 비교할 실험 계열의 안정된 이름이다. dataset·split·목적을 담고 Windows 금지문자, 공백과 끝의 마침표를 피한다. 한글과 underscore는 허용한다. 동일 종류의 기존 문서를 전체 날짜에서 검색해 최대 attempt+1로 회차를 부여한다. 회차는 학습 round/epoch와 다르다. 종류를 바꾸거나 이름을 다르게 표기해 회차가 초기화되지 않도록 기존 이름을 재사용한다.

새 실행은 새 회차로 저장한다. 동일 실행의 진행 중 결과 추가/오기 정정은 기존 회차 문서를 갱신하고 변경 이유·시각을 남긴다. 중복 값을 덧붙이지 않는다. 실패·중단·관측 없는 실행도 상태와 빈 measurements로 기록하며 다음 회차에서 덮어쓰지 않는다. 기존 파일이 있으면 실행 동일성을 확인하고 임의로 덮어쓰지 않는다.

## 문서 내용과 구조화된 수치
[형식 규약](references/result-schema.md)을 읽고 [템플릿](assets/experiment-result-template.md)을 복사한다. UTF-8 Markdown에 한 개의 `experiment-results` fenced block과 유효한 JSON을 넣는다. JSON이 추출 기준이며 사람이 보는 요약표는 같은 값에서 만든다.

조건, 목적, 결과 해석, 근거 링크, 미확인과 다음 확인을 prose로 적는다. 숫자는 문자열·단위·퍼센트 기호·평균±표준편차와 섞지 않고 각 measurement에 분리한다. 값 0과 미측정 null을 구분한다. ratio/percent/percentage_point 단위를 명시하며 원시 정밀도를 보존한다. seed별 값, 평균·표준편차·표본 수는 별도 기록한다. best ROC round AP와 독립 best AP, final, round별 값을 구분한다. 없는 seed·round·run ID·수치를 만들지 않는다.

조건이 다른 비교군은 measurement별 conditions에 override하고 method·dataset·split·selection·statistic·source를 남긴다. 원시 로그가 없는 보고 수치는 source와 provenance=reported로 표시한다. 통계 계산이면 provenance=derived와 입력 근거를 남긴다. 요약·표·로그의 모순을 숨기지 않는다.

실제 문서를 저장하면 docs/experiments/README.md의 목록에 날짜·종류·회차·상태·링크를 추가하거나 갱신한다. 다른 목적지라면 해당 색인을 따른다. 현재 템플릿 예시는 실제 실험 기록으로 등록하지 않는다.

## 검증과 수치 추출
저장 후 단일 파일 `--check`로 JSON 형식·파일명·중복·수치를 검사한다. 필요하면 계열별 또는 전체 문서에서 CSV를 만든다.

```powershell
python .agents/skills/record-experiment-results/scripts/extract_results.py docs/experiments/2026-10-08_cctv_event_pel_weight_run001.md --check
python .agents/skills/record-experiment-results/scripts/extract_results.py docs/experiments --type cctv_event_pel_weight --output artifacts/experiment-metrics/cctv_event_pel_weight.csv
```

CSV는 측정당 한 행인 long 형식이다. 빈 값은 미측정이며 0으로 치환하지 않는다. 원본 문서 경로, 회차·조건·metric·unit·seed·round·statistic·selection이 같이 나와 비교 조건을 추적할 수 있다. 기본은 기존 CSV 덮어쓰기 금지이며 갱신 요청이면 `--overwrite`를 쓴다. 출력 인자가 없으면 stdout에 CSV를 출력한다. 오류 문서는 건너뛰지 않고 실패하므로 고친 뒤 다시 추출한다.

CVPR 표나 이미지를 요청하면 저장한 측정값을 기존 export-cvpr-results 스킬의 입력으로 사용한다. 지식·원인 가설의 상세 발표용 설명은 write-seminar-notes와 연결한다. 이 스킬만으로 표 이미지나 발표 문서를 자동 생성하지 않는다.

완료 보고에 개별 기록·회차와 통합표 링크, 두 파일의 동기화·검증 결과 및 요청한 경우 CSV 링크를 제공한다. 통합표 갱신 실패를 개별 기록만으로 완료 보고하지 않는다.

## 매 호출 시 통합 비교표 함께 갱신
결과를 추가하거나 정정하는 스킬 호출에서는 개별 회차 문서와 `docs/experiments/results-summary.md`를 반드시 함께 갱신한다. 다른 기록 디렉터리를 지정한 경우에도 그 디렉터리에 results-summary.md 한 문서를 유지한다.

개별 JSON을 먼저 검증하고 아래 도구로 전체 기록에서 통합표를 재생성한다. 단순 append 대신 전체 재생성으로 정정·추가가 정확히 반영되고 중복 행이 누적되지 않게 한다. 원본 JSON은 유지하고 통합표를 수동 편집하지 않는다. 새 실제 기록이 없을 때는 숫자를 만들지 않는다.

```powershell
python .agents/skills/record-experiment-results/scripts/update_summary.py docs/experiments
python .agents/skills/record-experiment-results/scripts/extract_results.py docs/experiments --check
```

통합 문서의 한 표에 날짜순으로 값을 배치하고 metric/unit별 수치 열을 만든다. 날짜 종류·실험 종류·회차·방법·dataset·split·seed·학습 round·통계·선택 정책·조건·출처·개별 문서 링크를 유지한다. 같은 조건의 지표만 같은 행에 두며 다른 조건이나 단위는 합치지 않는다. 원시 정밀도와 0을 보존하고 미측정은 --로 표시한다. 측정이 없는 실패/진행 중 회차는 별도 상태 목록에 남긴다.

생성된 통합 문서는 CSV 추출 대상에서 제외한다. CSV와 통합표 모두 개별 JSON을 원본으로 사용한다. 갱신 후 원본 수치·단위·날짜 정렬·링크와 한글을 확인한다. 도구는 모든 원본 검증 성공 후 통합표를 교체하며 오류 시 이전 통합표를 보존한다. 실패하면 원본을 수정해 다시 갱신하고 남은 문제를 명시한다.

## 공식 용어와 축약 표현 규칙
문서를 생성·정리·갱신할 때 [공통 용어 지침](../../../docs/writing-guidelines.md)을 반드시 따른다. 논문·명세·공식 문서에서 확인한 공식 용어와 공식 약어만 사용한다. 어시스턴트가 자체적으로 만든 축약 표현은 사용하지 않는다. 공식 명칭이 없는 실험·변형은 대상·방법·조건·목적이 이해되도록 풀어서 쓰며, 조사 생략과 단어 나열은 의미가 분명한 경우에만 허용한다.

제목·파일명·표의 행과 열·통합표·세미나 문서·도식 label·수식 설명·완료 보고에도 동일하게 적용한다. 코드 식별자·원본 로그는 정확히 인용하되 독자용 완전한 설명을 병기한다. 기존 기계 판독 필드를 임의 변경하지 않는다. 기존 스킬의 짧은 예시보다 이 규칙이 우선하며 저장 전에 공식성·명확성을 확인한다.
