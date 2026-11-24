# 실험 수치 형식 v1

문서에 정확히 한 개의 experiment-results fenced block을 넣고 내부는 표준 JSON으로 쓴다. 주석·trailing comma는 허용하지 않는다. YAML frontmatter나 Markdown 표를 수치 추출의 원본으로 사용하지 않는다.

## 문서 필드
| 필드 | 형식 | 의미 |
| --- | --- | --- |
| schema_version | 정수 1 | 형식 버전 |
| date | YYYY-MM-DD | 파일명 날짜 |
| date_kind | experiment 또는 report | 날짜의 근거 |
| experiment_type | 문자열 | 파일명의 실험 계열 |
| attempt | 양의 정수 | 파일명 runNNN과 일치. 전체 날짜에서 같은 계열마다 증가 |
| track | 문자열 | pel / aggregation 등 실제 담당 트랙 |
| status | planned/running/completed/failed/interrupted/reported | 실행 또는 보고 상태 |
| conditions | 객체 | 공통 설정. split·집계·optimizer·개인화·예산·특징·평가 정책·commit·run ID 등. 미확인은 null |
| measurements | 배열 | 측정값. 아직 없으면 [] |

## measurement 필드
모든 행은 아래 필드를 명시한다.

| 필드 | 형식 | 의미 |
| --- | --- | --- |
| method/dataset/split | 비어 있지 않은 문자열 | 측정 대상과 평가 조건 |
| metric | 문자열 | roc_auc/ap/loss 등 통일된 이름 |
| value | 유한 숫자 또는 null | 단위 없는 원시 숫자. 미관측 null, 실제 0은 0 |
| unit | ratio/percent/percentage_point/count/ms/seconds/bytes/unitless | 단위. ratio는 0–1, percent는 0–100 |
| seed | 정수 또는 null | seed 미확인·집계 통계는 null |
| round | 0 이상 정수 또는 null | 학습 라운드. 회차 attempt와 별개. 초기 평가 round=0 가능 |
| statistic | single/mean/std/delta/n | 관측·통계 유형 |
| selection | 문자열 | final/best_roc_round/independent_best_ap/per_round/validation_selected 등 구분 |
| provenance | observed/reported/derived | 원시 관측/보고 전사/통계 계산 |
| source | 문자열 | 로그 상대 경로 및 행/보고 페이지/계산 입력. 수치가 있으면 필수 |
| conditions | 객체 | 공통 설정과 다른 값. 없으면 {}. batch·epoch·client ID·weight·개인화 여부 등 |

행 식별자는 method, dataset, split, metric, unit, seed, round, statistic, selection, conditions로 구성된다. 동일 관측이 보고와 로그에 중복되면 한 행과 본문 출처 대조로 정리한다. 서로 다른 run/epoch/client는 conditions로 구분한다.

측정 조건은 공통 conditions에 행 conditions를 덮어쓴 유효 설정으로 CSV에 출력한다. metric/unit/selection 이름은 기존 계열에서 유지한다. 다른 값으로 이름을 바꾸거나 단위를 암묵 변환하지 않는다. 표시용 반올림은 설명표에서만 적용한다.

mean/std/n은 별도 행이다. n의 unit은 count, mean/std는 대상 metric과 동일 단위다. std가 sample인지 population인지 conditions.std_definition에 남긴다. delta는 baseline 행/실행을 source와 conditions에 연결하고 퍼센트 차이면 percentage_point로 명시한다. round 간 변동을 seed std로 기록하지 않는다.

extract_results.py는 구조, 이름/날짜/회차 일치, 유한 값, ratio/percent 범위(관측/평균), 중복 측정을 검증한다. 결과의 과학적 타당성·출처 존재·비교 공정성은 코드와 로그를 대조해야 한다.


## 파생 통합 문서
results-summary.md는 전체 회차의 JSON에서 생성하는 날짜순 수치 비교표다. 동일 조건의 metric/unit 열을 합치고 원시 값·미측정·출처를 보존한다. README.md와 results-summary.md는 CSV와 통합표 입력 기록에서 제외한다. 개별 문서 정정 뒤 update_summary.py로 전체 표를 다시 생성한다.
