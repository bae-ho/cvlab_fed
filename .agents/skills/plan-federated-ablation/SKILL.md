---
name: plan-federated-ablation
description: fed-wavad의 집계·개인화·프롬프트 토큰·패딩 어블레이션을 설계하거나 결과를 검토하고 연구 주장에 필요한 평가 조건을 점검한다.
---

# 연합학습 어블레이션

[회의 근거](../../../docs/research/meeting-evidence.md), [구현 상태](../../../docs/research/implementation-status.md)를 읽고 요청에 해당하는 코드만 확인한다. FedPGP 주 후보, FedProx/FedDyn 비교, Scaffold 당시 제외는 회의 우선순위이며 사용자 요청의 대체물이 아니다. 미구현 FedDyn 명령을 제시하지 않는다.

실제 서버 선택을 train.py에서 추적한다. algorithm, aggregation_method, server_optimizer, personalization_type, local_param_policy와 두 contrastive 옵션을 각각 기록한다. FedPGP에서는 공유 가중치 평균뿐 아니라 로컬 어댑터 보존과 글로벌 배포를 검증한다.

집계만 비교할 때 optimizer·개인화·손실·학습 예산을 통제한다. FedAvg 대 FedPGP처럼 어댑터 유무가 같이 달라지면 결합 효과를 보고하고 추가 대조군의 필요성을 설명한다. 사용자가 지정하지 않은 전체 탐색을 자동 실행하지 않는다.

n_ctx 4/8/16은 prompt_net.py의 생성 토큰, model.py의 X 자리 수·클래스/EOT 위치, 어댑터 shape를 연결해서 설계한다. prefix/postfix는 다른 설정이다. 모듈 위치 비교는 후속 안건이며 이미 검증된 구조로 단정하지 않는다.

패딩 비교는 0 채움, 학습/LLM 특징 채움, attention mask를 별도 축으로 다룬다. 유효 길이가 attention/PEL pooling에 전달되는지 확인한다. 완전 마스킹 행의 NaN 및 실제 연산 효율을 구분한다.

[검증 기준](../../../harness/checks/research-validation.md)으로 특징·GT 정렬, 누수, batch/step 수, 평가 분할을 점검한다. unseen은 전체 학습 클래스 노출, domain gap은 로컬 held-out/FPR threshold, 실시간은 종단간 범위, 프라이버시는 전달 데이터 흐름의 근거가 필요하다.

[실험 템플릿](../../../harness/templates/research-experiment.md)에 비교표, 통제 변수, 실행 가능 여부와 코드상 선행 작업을 기록한다. 로그 검토는 실제 관측과 가설을 분리한다. 시드/테스트 기반 checkpoint 선택 등의 한계도 명시한다.

## 트랙별 비교 경계
[두 트랙 범위](../../../docs/research/two-track-scope.md)를 우선한다. 현재 PEL 작업은 집계·서버 optimizer·개인화 정책을 고정한 PEL 대조군 설계다. 집계 변경·조합 탐색과 통합은 후속 별도 범위다. 위 집계 비교 절차는 집계 트랙의 실험 설계를 요청받은 경우에만 적용한다.

scene/event에서 기준과 후보의 클라이언트·데이터·예산을 유지한다. 로컬 클래스 구성과 실제 denominator를 확인하고 다른 클라이언트 데이터를 합치지 않는다. CCTV scene 미준비를 명시한다.

[Slack 조합](../../../docs/research/slack-experiment-evidence.md)은 momentum 포함/제외를 분리한다. 제외 표의 하락 split, seed·비용 미검증을 보존한다. 현행 SOTA 목표는 달성 사실과 구분하며 unseen·실시간 등을 자동 필수 항목으로 추가하지 않는다.

## 공식 용어와 축약 표현 규칙
문서를 생성·정리·갱신할 때 [공통 용어 지침](../../../docs/writing-guidelines.md)을 반드시 따른다. 논문·명세·공식 문서에서 확인한 공식 용어와 공식 약어만 사용한다. 어시스턴트가 자체적으로 만든 축약 표현은 사용하지 않는다. 공식 명칭이 없는 실험·변형은 대상·방법·조건·목적이 이해되도록 풀어서 쓰며, 조사 생략과 단어 나열은 의미가 분명한 경우에만 허용한다.

제목·파일명·표의 행과 열·통합표·세미나 문서·도식 label·수식 설명·완료 보고에도 동일하게 적용한다. 코드 식별자·원본 로그는 정확히 인용하되 독자용 완전한 설명을 병기한다. 기존 기계 판독 필드를 임의 변경하지 않는다. 기존 스킬의 짧은 예시보다 이 규칙이 우선하며 저장 전에 공식성·명확성을 확인한다.
