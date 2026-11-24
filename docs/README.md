# 프로젝트 지식 지도

- [연구 코드·실험 결과 인수인계](HANDOVER.md): 후배 연구원의 PEL 구현과 현재 구현 비교, 실행 환경, 결과 해석과 후속 작업

## 현행 연구 목표와 성능 정의

- 최종 목표: **연합학습 환경의 약지도 비디오 이상 탐지 모델 중 비교 가능한 벤치마크 설정에서 SOTA 달성**.
- 개발 방향: Fed-WSVAD에 PEL4VAD의 전경·배경 분리 및 의미 정렬을 결합하여 기존 벤치마크 탐지 성능을 높인다.
- 직접적인 개선 목표: Fed-WSVAD 기준 모델을 초과하고, 중앙집중 학습 PEL4VAD의 성능도 상회하는 결과를 목표로 한다. 중앙집중 PEL4VAD를 초과하는 것과 연합학습 SOTA를 달성하는 것은 별도로 판단한다.
- 필수 범위: 통상적인 학습·테스트 분할에서 프레임 단위 VAD 성능 평가. 미학습 이벤트, zero-shot, leave-one-event-out 검증은 현재 성공 기준에 포함하지 않는다.
- 상태: 목표 정의 완료. SOTA 달성 여부는 미검증이다.

| 평가 대상 | 주 지표 | 성능 해석 |
| --- | --- | --- |
| UCF-Crime | 프레임 단위 ROC-AUC | 기준 모델 및 동일 연합 분할의 선행 방법과 비교 |
| XD-Violence | 프레임 단위 AP | 기존 코드의 average_precision_score 정의를 명시하고 비교 |
| AI Hub CCTV | 프레임 단위 ROC-AUC, 보조 AP | 자체 분할에서 개선을 검증. 이 결과만으로 공개 벤치마크 SOTA를 주장하지 않음 |

성능 보고에는 데이터·분할·특징 추출기·클라이언트 구성·학습 예산·seed·체크포인트 선택·점수 smoothing 조건을 함께 기록한다. 동일 설정 재현 비교와 서로 다른 설정의 논문 보고 수치는 분리한다. 여러 seed의 평균과 변동을 보고하고, validation에서 선택한 모델을 test에서 평가한다.

SOTA 비교군은 평가 시점의 연합학습 약지도 VAD 문헌을 확인하여 데이터셋과 분할별로 확정한다. Fed-WSVAD 한 방법만 초과했다고 SOTA로 판정하지 않는다. 이미 공개된 논문 수치를 목표 상한이나 현재 달성 수치로 복사하지 않는다.

2026-08 회의록과 과제 제안서의 신종 이상 탐지 목표는 역사적 배경이다. 현재 작업 범위는 2026-10-06 사용자 지시에 따라 위 정의로 갱신되었다. [목표 변경 결정](decisions/0002-vad-performance-objective.md)과 [회의 자료 안내](meeting-note/README.md)를 참고한다.

## 문서 탐색

- [프로젝트 실행 안내](../README.md)
- [코드 구조](architecture/README.md)
- [PEL 적용 분석과 이벤트 확장 판단](architecture/pel4vad-integration.md)
- [현행 목표의 결정 근거](decisions/0002-vad-performance-objective.md)
- [기존 회의 자료의 적용 범위](meeting-note/README.md)
- [기존 집계 문서](../AGGREGATION.md)
- [진행 중인 계획](plans/active/README.md)
- [완료된 계획](plans/completed/README.md)
- [설계 결정](decisions/README.md)
- [하네스 운영](../harness/README.md)

동일한 사실을 여러 곳에 복제하지 않고 원본 문서나 코드로 연결한다. 설정 기본값은 utils/config.py를 확인한다.

## 회의 기반 연구 지식
- [회의·제안서 근거와 열린 항목](research/meeting-evidence.md)
- [회의 요구와 구현 차이](research/implementation-status.md)
- [로컬 PEL + FedPGP 결정 기록](decisions/0001-local-pel-fedpgp.md)

## 두 트랙과 현재 작업 범위
장기태 연구원은 가중치 합산 개선, 김배호 연구원은 CCTV 적용과 PEL4VAD 결합을 각각 실험 중이다. 사용자는 김배호 연구원의 적용점과 실험 설계를 지원하며 현재 집계 방식을 변경하지 않는다. 가장 좋은 방법들의 통합은 이후 진행한다.

scene/event는 지자체별 클라이언트를 고려한 실험 분할이다. PEL 적용이 두 분할의 데이터·대조학습 조건 및 향후 집계 인터페이스와 충돌하지 않도록 확인한다.

- [두 트랙 범위와 분할 호환성](research/two-track-scope.md)
- [작업 경계 결정](decisions/0003-two-track-pel-scope.md)
- [Slack 실험 근거 대조](research/slack-experiment-evidence.md)


- [세미나용 지식 노트](seminar/README.md): 설계·실험 과정의 개념, 아키텍처와 흐름을 발표용 Markdown으로 정리한다.

- [회차별 실험 결과](experiments/README.md): 설명·조건·원시 수치를 함께 기록하고 CSV로 추출한다.

모든 문서 생성·정리·갱신에는 [공통 용어 지침](writing-guidelines.md)을 적용한다. 공식 용어와 공식 약어만 사용하고, 나머지는 의미를 알 수 있게 풀어서 쓴다.
