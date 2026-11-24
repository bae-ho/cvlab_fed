# 실험명

- 날짜 / 코드 커밋 / 실행자:
- 관련 요구 ID / 원본 페이지:
- 상태: 계획 / 동작 확인 / 성능 확인 / 미검증
- 가설과 성공 기준:

## 실행 전 설명과 설계
- 사용자에게 설명할 실험 전체 이름 (파일·스크립트 약칭과 구분):
- 왜 지금 실행하는가 / 선행 결과와 남은 질문:
- 비교군 / 이번에 변경하는 요인 / 고정하는 요인:
- 무엇을 측정하며 어떤 관측으로 가설을 지지·반박하는가:
- 결과로 판단할 수 없는 내용 / 후속 확인:
- 사전 설명·문서화 완료 시점 / 실행 상태 / 자원 및 예상 비용:

## 통제 조건 상세
- 데이터 버전 또는 manifest 식별값 / 특징 추출기 / 차원 / stride:
- train/validation/test 및 클래스·도메인 분할 / 누수 확인:
- 초기 checkpoint / 시드 설정 범위 / 결정성 제한:
- normal/anomaly batch 수 / 실제 local steps:
- 모델 / n_ctx / prefix/postfix / padding 정책:
- algorithm / 실제 서버·클라이언트 / aggregation_method / server_optimizer:
- personalization_type / local_param_policy:
- PEL 및 FedPGP contrastive 옵션 / weight / temperature:
- rounds / epochs / batch size / optimizer / LR:
- 장치 / 라이브러리 환경:

## 비교와 결과
| 조건 | ROC-AUC (0–1) | AP (0–1) | 기준 대비 delta | 로그 / checkpoint |
| --- | --- | --- | --- | --- |

## 검증 기록
- 실제 실행 명령 / 실행 시간 / 종료 상태:
- loss·gradient·파라미터 업데이트 확인:
- 평가 범위: global / global proxy with local adapter / local held-out / unseen:
- checkpoint 선택 정책 / test 재사용 여부:
- GT 정렬 및 길이 확인:

## 결론
동작 여부와 성능 유효성을 따로 기록한다. 주장 가능한 범위, 미검증 항목, 다음 실험을 적는다.

## 트랙과 호환성
- 담당 트랙 / 담당 연구원 / 이번 변경 범위:
- 고정 집계 방식·가중치·서버 optimizer·공유/로컬 정책 / 보존 근거:
- 분할: scene / event / 별도 진단 실험. 지자체 클라이언트 맥락과 데이터 소유 경계:
- 로컬 실제 클래스 / 배치 후보 / 텍스트 인덱스 / 특징·GT·유효 길이:
- 새 모듈 gradient와 shared/local 상태 key·shape / 추후 통합 영향:

## 보고와 안정성
- 보고 날짜 / 물리 페이지 / 표 ID / run ID / 설정 출처 / 불일치:
- seed 목록 / seed별 delta / 평균·표준편차 / 표본 수:
- 최고 ROC / 해당 round AP / 최고 round / 독립 최고 AP 및 round / 최종 ROC·AP:
- 전체 round 곡선 / 정체·하락 / loss·gradient·update 근거:
- trainable/total params / 연산 입력·단위 / 메모리 / latency·throughput / 미측정 항목:
- 보고 완료 / 로그 확인 / 재현 확인 / 후속 계획을 각각 표시:

## 작성 표현 확인
- 논문·명세·공식 문서에 있는 공식 용어 및 공식 약어만 사용한다. 자체적으로 만든 축약 표현은 쓰지 않는다.
- 실험 대상·방법·조건·목적을 풀어서 적는다. 의미가 분명할 때만 조사를 생략하고 단어를 나열한다.
- 코드 식별자는 정확히 인용하고 완전한 설명을 병기한다. 제목·표·도식·수식 설명에도 동일하게 적용한다.
