---
name: validate-local-pel
description: fed-wavad의 단일 로컬 PEL 대조학습을 MIL 기반 분류 손실 기준과 비교해 동작 및 성능을 따로 검증하거나 기존 실험 로그를 해석한다. CCTV/UCF 로컬 PEL 비교 요청에 사용한다.
---

# 단일 로컬 PEL 검증

[코드 상태](../../../docs/research/implementation-status.md), [검증 기준](../../../harness/checks/research-validation.md)의 단일 로컬 항목, scripts/experiment_pel4vad_client.py와 utils/tools.py를 읽는다. 이식된 스타일 손실과 원논문 전체 재현은 구별한다.

로그 분석 요청이면 기존 logs/와 runs_4090/에서 dataset, client, baseline, pel4vad, delta, 설정을 찾는다. 로그에 없는 시드·성능·학습 예산을 채우지 않는다. 실행이 요청된 경우 데이터와 의존성, batch 수, 장치 및 자원 범위를 확인한다.

기존 스크립트의 선택 가능한 클래스와 옵션을 확인해 명령을 만든다. 예시: `python scripts/experiment_pel4vad_client.py --dataset cctv --client-class Fight --epochs 10 --seed 88888888`. 횟수와 장치는 사용자가 지정한 범위에 맞춘다. 기준과 후보의 설정을 통제한다. 이 스크립트는 torch 시드만 설정하므로 완전 결정성은 보장하지 않는다.

동작은 유한 loss/gradient, 클래스 인덱스와 shape, 실제 업데이트, 유효 길이 영향을 확인한다. 성능은 ROC-AUC/AP와 delta를 따로 기록한다. 현재 shared/global test set 비교는 proxy이며 로컬 도메인 또는 strict unseen 검증이 아니다.

연합 통합 요청이면 train.py의 실제 서버 경로와 PEL 인자 전달을 먼저 추적한다. 기본 FedAvg의 전달 누락은 수정되었다. `legacy`와 `fixed_bank`를 구별하고 [새 실행 안내](../../../docs/research/fixed-bank-pel-run.md)의 gradient/denominator 검증을 적용한다. XD 다중 라벨을 이 단일 라벨 절차에 그대로 적용하지 않는다.

[실험 템플릿](../../../harness/templates/research-experiment.md)으로 결과·명령·로그·평가 범위·미검증을 남긴다. 실행 불가 시 검증된 코드 사실과 실행에 필요한 조건까지 기록하고 성능 수치를 생성하지 않는다.

## 현재 트랙과 scene/event 조건
[두 트랙 범위](../../../docs/research/two-track-scope.md)를 먼저 읽는다. 현재는 김배호 연구원의 PEL 적용점과 설계를 지원하며 집계·서버 optimizer·공유/로컬 정책을 유지한다. PEL 옵션 미전달을 다른 집계 방식으로 우회하지 않는다.

[PEL 분석](../../../docs/architecture/pel4vad-integration.md)에서 시각/텍스트 특징·손실·gradient·실제 대조 후보를 확인한다. event 단일 이상 클래스 로컬과 scene 실제 배치 구성을 구분한다. 다중 이벤트 로컬 진단은 별도 실험이며 클라이언트 데이터 합침이나 event/scene 교체를 해결책의 전제로 삼지 않는다. CCTV scene은 아직 준비되지 않았다. 새 상태 key/shape와 공유 gradient를 기록해 후속 통합 호환성을 확인한다.

[Slack S05](../../../docs/research/slack-experiment-evidence.md)의 best/final, 해당 round AP/독립 최고 AP를 구분한다. round 변동은 seed 분산이 아니다. Fight 또는 CCTV 정체만으로 PEL 전체를 기각하지 않는다. unseen은 현행 필수 범위에 자동 추가하지 않는다.

## 공식 용어와 축약 표현 규칙
문서를 생성·정리·갱신할 때 [공통 용어 지침](../../../docs/writing-guidelines.md)을 반드시 따른다. 논문·명세·공식 문서에서 확인한 공식 용어와 공식 약어만 사용한다. 어시스턴트가 자체적으로 만든 축약 표현은 사용하지 않는다. 공식 명칭이 없는 실험·변형은 대상·방법·조건·목적이 이해되도록 풀어서 쓰며, 조사 생략과 단어 나열은 의미가 분명한 경우에만 허용한다.

제목·파일명·표의 행과 열·통합표·세미나 문서·도식 label·수식 설명·완료 보고에도 동일하게 적용한다. 코드 식별자·원본 로그는 정확히 인용하되 독자용 완전한 설명을 병기한다. 기존 기계 판독 필드를 임의 변경하지 않는다. 기존 스킬의 짧은 예시보다 이 규칙이 우선하며 저장 전에 공식성·명확성을 확인한다.
