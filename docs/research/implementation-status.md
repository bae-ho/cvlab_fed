# 회의 요구와 현재 구현의 차이

2026-10-06 코드 및 CPU 합성 검증 기준. 실제 데이터 학습·성능 비교는 실행하지 않았다. 새 PEL의 구현과 명령은 [고정 전체 클래스 PEL 실행](fixed-bank-pel-run.md)을 따른다.

현행 성능 목표는 [프로젝트 성능 정의](../README.md#현행-연구-목표와-성능-정의)를 따른다. PEL의 공식 구현 비교, 기존 정량 로그와 다중 이벤트 확장 조건은 [PEL 적용 분석](../architecture/pel4vad-integration.md)에서 관리한다.

| 요구 | 코드에서 확인한 상태 | 남은 검증 또는 차이 |
| --- | --- | --- |
| R01/R02 로컬 PEL | 기존 standalone/legacy 손실을 유지하고 기본 FedAvg에 fixed-bank PEL 추가 | 기존 스크립트는 이벤트 하나와 Normal 및 동적 텍스트. 새 변형은 전체 고정 텍스트 후보와 시각 경로 전용 gradient 사용. CPU 동작 검증 완료, 실제 성능 비교 미실행 |
| R04 FedPGP | fed/server.py에서 공유 파라미터를 클라이언트가 반환한 DataLoader 배치 수로 가중 평균; utils/model.py에서 로컬 파라미터 분리; train.py가 fedpgp + none을 low_rank로 변환 | 실제 가중치는 원시 표본 수와 다를 수 있음. 로컬 어댑터 보존, 공유 가중치 배포, 개인화 추론을 함께 검증 |
| R04 비교 후보 | config에 fedavg/fedpgp/fedprox/scaffold/fednova 및 서버 optimizer none/momentum/adam/yogi 존재 | FedDyn 옵션은 없다. Scaffold는 구현되어 있으나 당시 회의 주 실험 후보에서 제외. 코드 삭제나 사용 금지로 해석하지 않음 |
| R03 토큰 4/8/16 | utils/prompt_net.py의 n_ctx=4와 utils/model.py의 클래스 문자열 X 4개 하드코딩 | 전용 n_ctx CLI 없음. prompt_prefix/postfix는 n_ctx와 다름. 두 하드코딩과 어댑터 shape/텍스트 위치를 함께 조정해야 함 |
| R06 패딩 | 새 fixed-bank PEL pooling은 유효 길이 mask 적용 | legacy PEL과 기존 Temporal/프롬프트 패딩 동작은 유지. 전체 모델의 패딩 불변성을 주장하지 않음 |
| R07 CCTV | scripts/prepare_cctv_dataset.py가 CLIP ViT-B/16 특징, CSV/GT/manifest 생성; utils/dataset.py가 CCTV event 6클래스 지원 | M14의 I3D 기반 PEL 준비와 구별. 특징 차원·추출기·stride가 다른 파일을 섞지 않음. CCTV scene/random은 현재 준비되지 않음 |
| R08 unseen | 기존 공통 테스트셋 ROC/AP 평가 | 현재 필수 평가 범위에서 제외. 향후 해당 주장을 할 때만 별도 검증하며 기존 전체셋 결과를 unseen 증거로 해석하지 않음 |
| R09 domain gap | personalized 모델도 공통/global test split으로 평가 | 로컬 도메인 held-out 평가가 아니다. 도메인 개선 주장에 별도 분할 필요 |
| R10 프라이버시·실시간 | 로컬 학습 시뮬레이션과 가중치 집계 | 네트워크 데이터 반출 통제, 보안 보장, 엣지 종단간 지연을 검증한 근거는 확인하지 않음 |

## 실행 경로에서 놓치기 쉬운 점
- train.py에서 algorithm='FedAvg' 분기 안에서 실제 서버를 선택한다. aggregation_method와 server_optimizer/local_param_policy까지 확인한다. algorithm의 다른 선택지 노출만으로 실행 가능하다고 판단하지 않는다.
- 기본 FedAvg도 PEL 인자를 직접 전달한다. `legacy`는 기존 동적 텍스트 손실, `fixed_bank`는 전체 고정 기준의 변형이다. 기본은 PEL off. 2026-10-08 사용자 요청으로 강도·시점·pooling·gradient 범위·projection 및 명시적 FedNova/FedPGP 연결 옵션을 추가했다. integration 옵션 없이 plain FedAvg 제한 유지; 기존 집계 식·정책 보존. [후속 계획과 CPU 검증](../plans/active/2026-10-08-pel-federated-compatibility.md), 실제 새 후보 학습 미실행.
- train.py는 `--seed`를 제공하고 기본값 88888888을 유지한다. bank 생성은 RNG와 CLIP 모드를 보존한다. 단일 로컬 스크립트의 torch-only seed는 유지하며 완전 결정성은 보장하지 않는다.
- os.makedirs로 save 상위 경로도 생성한다. 기본 FedAvg에서 manifest, round metrics, best/final checkpoint와 summary를 저장한다.
- checkpoint는 서버에 따라 client_*_personalized.pth 또는 client_*_local.pth로 저장된다. 재개/추론 로더가 기대하는 이름과 공유·로컬 가중치 시점을 맞춘다.
- 서버에서 공통 테스트셋 성능으로 best checkpoint를 선택한다. 최종 주장에는 선택용 validation과 최종 test 분리가 필요하다.
- inference.py와 로컬 실험은 스니펫 점수를 16번 반복해 GT와 비교한다. 추출 stride, 샘플 순서, 길이 일치를 먼저 검증한다.

관련 코드: [학습](../../train.py), [서버](../../fed/server.py), [클라이언트](../../fed/client.py), [모델](../../utils/model.py), [토큰](../../utils/prompt_net.py), [손실](../../utils/tools.py), [설정](../../utils/config.py), [로컬 비교](../../scripts/experiment_pel4vad_client.py), [CCTV 준비](../../scripts/prepare_cctv_dataset.py).

## 추가 보고와 현재 작업 경계
[Slack 결과](slack-experiment-evidence.md)는 집계 후보와 CCTV 정체의 기록이다. PDF 실행 커밋·seed·원시 로그 연결이 불완전하며 seed·비용 검증 완료로 해석하지 않는다.

[두 트랙 범위](two-track-scope.md)에 따라 현재는 집계를 고정하고 PEL 적용점·손실·연결 경로를 검토한다. FedAvg의 PEL 옵션 미전달은 집계 교체로 우회하지 않는다. scene/event의 로컬 클래스 구성과 텍스트 후보·gradient·유효 길이를 확인한다. CCTV는 현재 event만 준비되어 있다.
