# PEL4VAD 분석과 연구 목표 문서화

- 날짜: 2026-10-06
- 상태: 완료

## 요청과 성공 기준

- 기존 하네스에 코드 분석, 기존 결과, 적용 후보와 이벤트 확장 판단을 기록한다.
- 현행 성능 목표를 연합학습 환경의 약지도 VAD SOTA 달성으로 정의한다.
- 기존 PEL4VAD 및 Fed-WSVAD보다 높은 벤치마크 성능을 목표로 하며 미학습 이벤트 검증은 필수 조건에서 제외한다.
- 기존 목표가 있는 문서를 확인하고 현행 정의와 역사 자료의 관계를 명시한다.

## 범위와 제약

- 문서 변경만 수행한다. 학습 코드, 데이터, 체크포인트와 기존 로그는 보존한다.
- 방법론 적용 조건을 정리하며 구체적인 실험 실행 계획은 후속 작업으로 남긴다.
- 성능 향상 가능성과 측정된 성능을 구분한다. SOTA는 목표이며 달성 사실로 기록하지 않는다.

## 관련 코드와 문서

- [하네스 운영](../../../harness/README.md)
- [프로젝트 지식 지도](../../README.md)
- [코드 구조](../../architecture/README.md)
- [회의 자료](../../meeting-note/)
- `scripts/experiment_pel4vad_client.py`, `utils/tools.py`, `fed/client.py`, `train.py`

## 실행 단계

- [x] 하네스 및 기존 목표 자료 확인
- [x] 공식 PEL 구현의 이벤트와 텍스트 후보 구성 확인
- [x] 문서 작성 및 기존 색인 갱신
- [x] 상대 링크와 코드 설명 검증
- [x] 결과를 기록하고 completed로 이동

## 검증 기록

| 명령 또는 확인 | 결과 | 근거 경로 |
| --- | --- | --- |
| Markdown 목표 검색 및 PDF 텍스트 추출 | Markdown 성능 정의 없음. 회의록과 제안서에 미학습 이벤트 목표 있음 | `docs/meeting-note/` |
| 공식 PEL `dataset.py`, `utils.py`, `train.py` 확인 | 샘플별 두 텍스트를 배치에서 결합 | [분석 문서](../../architecture/pel4vad-integration.md) |
| Python pathlib와 링크 추출로 Markdown 22개 검사 | 완료 이동 전 로컬 링크 93개 모두 존재, UTF-8 정상 | `docs/`, `harness/`, `AGENTS.md` |
| 코드 문자열 assertion | 실제 PEL 텍스트 선택, 단일 class index 고정, legacy FedAvg 옵션 누락 확인 | `utils/tools.py`, `scripts/experiment_pel4vad_client.py`, `train.py` |
| 목표와 unseen 문구 검색 | 현행 목표를 단일 원본에 정의하고 역사 자료에 변경 관계 연결 | `docs/README.md`, `docs/research/`, `harness/checks/` |
| `git diff --check` | 통과. 문서가 현재 untracked이므로 이 명령만으로 내용 검증을 대체하지 않음 | 작업 디렉터리 |

## 결과와 남은 문제

기존 `docs/README.md`에 현행 성능 목표를 정의하고, 코드 구조·분석·결정·연구 근거·검증 기준을 연결했다. 작업 중 추가된 연구 하네스 문서를 보존하며 현행 목표와 충돌하는 상태 설명을 갱신했다. 기존 ADR 0001을 보존하고 새 결정에는 0002를 사용했다.

사전 문자열 추가와 실제 다중 이벤트 시각 학습의 차이, 원본의 배치 기반 후보 구성, 적용 후보를 문서화했다. 목표는 정의했지만 성능 달성은 검증하지 않았다. 코드 변경, 데이터와 체크포인트를 이용한 학습·추론, 세부 실험 실행 설계와 Git commit은 수행하지 않았다.

환경: Windows PowerShell, Python 3.13.5. PDF 확인에는 pypdf를 사용했다.
