# PEL 의미 정렬을 결합한 연합학습 비디오 이상 탐지 연구 인수인계

- 기준일: 2026-10-21
- 전달 버전 태그: `handover-2026-10-21`
- 범위: 코드·설계·실험 결과·진단 해석·남은 작업 전달
- 이번에 하지 않은 작업: 데이터와 모델 가중치의 분리·이동·추가 배포, 새 학습

## 1. 연구 목적과 현재 판단

목적은 연합학습 환경에서 비디오 이상 탐지 성능을 높이는 것이다. 현재는 MIL 기반 분류 손실에 PEL4VAD에서 착안한 전경·배경 의미 정렬 손실을 추가하고, 이벤트별·장면별 클라이언트 분할에서 효과와 한계를 비교한다. 최종 목표는 비교 가능한 연합학습 비디오 이상 탐지 벤치마크에서 최고 성능 달성이다. **최고 성능 달성, 원본 PEL4VAD 성능 초과, 미학습 이벤트 탐지 효과는 아직 검증하지 않았다.**

- [현행 연구 목표와 성능 정의](README.md#현행-연구-목표와-성능-정의): 목표의 원본
- [두 연구 트랙과 작업 경계](research/two-track-scope.md): PEL 적용과 집계 개선은 독립 트랙; 현재 PEL 비교의 집계 수식·계수·서버 optimizer·공유/로컬 정책 유지
- [용어 지침](writing-guidelines.md): 공식 명칭 사용, 실험 이름은 데이터·분할·방법·목적을 풀어서 설명

미학습 이벤트나 zero-shot 검증은 현재 필수 성능 기준이 아니다. 새 실험을 제안할 때는 먼저 이유·설계·비교군·측정·해석 기준을 문서화하고 설명한 뒤 실행한다.

## 2. 먼저 읽을 문서

| 순서 | 문서 | 읽을 목적 |
| --- | --- | --- |
| 1 | [기존 네 조건의 실험 결과](research/2026-10-08-ucf-event-scene-results.md) | 이벤트별·장면별 분할에서 기준/PEL 성능 확인 |
| 2 | [날짜순 통합 측정값](experiments/results-summary.md)와 [실행별 색인](experiments/README.md) | 실행 조건·회차·원시 수치 추적 |
| 3 | [이벤트별 기준 모델 진단](experiments/2026-10-13_ucf_event_baseline_run002.md) | PEL 없는 업데이트 상쇄 기준 |
| 4 | [이벤트별 PEL 모델 진단 및 기준 비교](experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md) | 손실 gradient·client 업데이트 비교 |
| 5 | [손실 경쟁과 업데이트 상쇄 세미나 노트](seminar/2026-10-14_손실경쟁과_업데이트상쇄_진단.md) | 수식·측정 의미·해석 제한 |
| 6 | [전체 5단계 구현 설명](seminar/2026-10-08_PEL_연합학습_5단계_실험구현.md)와 [현행 계획](plans/active/2026-10-08-pel-federated-compatibility.md) | 후속 코드와 실행 완료 여부 구분 |

## 3. 후배 연구원의 기존 PEL 구현과 현재 구현의 차이

아래의 기존 구현은 저장소의 단일 클라이언트 스크립트와 `legacy` 손실을 기준으로 한다. 단일 실험은 Normal과 한 이상 이벤트를 학습하며, 이것을 여러 client의 FedAvg/FedPGP 효과 비교로 해석하지 않는다. 별도로 존재하는 연합 실행 경로·로그와도 구분한다.

| 비교 항목 | 후배 연구원의 기존 구현 | 현재 실제 비교 실험의 구현 |
| --- | --- | --- |
| 적용 위치 | Temporal 출력 시각 표현의 전경·배경 pooling에 PEL 정렬 추가 | 같은 시각 표현에 PEL 정렬 추가 |
| 학습 데이터 | 단일 클라이언트 스크립트에서 Normal과 지정 이벤트 하나 | UCF-Crime 이벤트별 13 clients 또는 장면별 9 clients; 원래 분할 유지 |
| 텍스트 기준 | 시각 문맥에 따라 생성되는 동적 텍스트 특징; 단일 실험은 두 클래스 | 모든 client가 공유하는 동결 CLIP 기반 14개 클래스 이름 특징; ConceptNet 확장 아님 |
| 실제 대조 후보 | batch에서 선택한 Normal/해당 이벤트 텍스트 쌍으로 구성한 정렬 행렬 | 각 시각 정렬 표본과 전체 14개 텍스트 클래스 사이의 행렬 |
| pooling 선택 점수 | 예측 logits에서 gradient 전달 | logits에서 분리한 점수 사용; PEL이 점수 경로를 직접 학습하지 않음 |
| 텍스트 정렬 대상 | 동적 프롬프트까지 PEL gradient 전달 가능 | 고정 텍스트 기준으로 PEL gradient 차단; MIL 분류 경로의 프롬프트 학습은 유지 |
| 유효 길이 | 기존 PEL pooling에 padding 제외 mask 없음 | PEL pooling에 유효 길이 mask 적용; 기존 Temporal 내부 padding 정책은 유지 |
| PEL이 학습시키는 대상 | 시각 모듈과 동적 텍스트·점수 경로 | 기본 후보는 Temporal·위치 임베딩; 별도 학습 가능한 PEL 파라미터 없음 |
| 서버 집계 | 단일 실험 스크립트에는 없음 | 기존 FedAvg 배치 수 가중 집계; 서버 optimizer·개인화·로컬 정책 none |
| 기본 검출 손실·추론 | MIL 기반 분류 손실, 동적 텍스트 기반 검출 | 동일 기본 분류 감독·추론 유지; 추론에 정답 이벤트 라벨을 사용하지 않음 |

클래스 이름을 14개로 늘리는 것과 ConceptNet으로 클래스별 의미 설명을 확장하는 것은 다른 변경이다. **현재 결과는 클래스 이름 기반 고정 기준 변형의 결과**다. 원본 PEL4VAD 전체를 재현했다고 표현하지 않는다.

event 분할은 한 client에 한 이상 이벤트와 Normal이 있는 구성이다. scene 분할은 한 client에 하나의 장면 범주와 여러 이벤트가 포함될 수 있는 구성이며, 동일 물리 카메라·장소임을 보장하는 지리적 식별 정보로 해석하지 않는다. PEL을 위해 client 간 영상을 합치거나 분할을 변경하지 않았다.

근거: [단일 클라이언트 스크립트](../scripts/experiment_pel4vad_client.py), [기존 손실](../utils/tools.py), [현재 고정 기준 손실](../utils/fixed_pel.py), [구조 비교 노트](seminar/2026-10-08_event와_scene_분할_및_PEL_적용비교.md). 기존 분석 문서의 작성 당시 “현재 구현”은 과거 구현을 뜻할 수 있으므로 날짜와 실행 경로를 함께 확인한다.

## 4. 확인된 성능과 해석

공통 조건: UCF-Crime, seed 88888888, 기존 FedAvg, 20 rounds × 10 local epochs, 정상/이상 loader별 batch 64, LR 0.0001. 표의 값은 백분율이며 최고 ROC-AUC 시점의 AP와 독립 최고 AP는 다른 선택이다.

| 클라이언트 분할 | 손실 조건 | 최고 ROC-AUC (%) | 독립 최고 AP (%) | 최종 ROC-AUC (%) | 최종 AP (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| 이벤트별 13 clients | MIL 분류 손실만 | 85.2737 | 28.6585 | 85.2338 | 26.6568 |
| 이벤트별 13 clients | MIL 분류 손실과 PEL 의미 정렬 | 86.1961 | 29.6821 | 83.9980 | 25.3017 |
| 장면별 9 clients | MIL 분류 손실만 | 84.6794 | 26.9744 | 83.8332 | 24.8047 |
| 장면별 9 clients | MIL 분류 손실과 PEL 의미 정렬 | 84.5076 | 27.2944 | 82.3111 | 24.0091 |

- 이벤트별 PEL은 최고 ROC-AUC +0.9224%p, 최종 ROC-AUC -1.2358%p. 최고값 개선을 안정적인 후기 개선으로 해석하지 않음
- 장면별 PEL은 최고 ROC-AUC 감소, 독립 최고 AP 증가; 모든 지표가 악화됐다는 해석과 구분
- 이벤트별 진단의 65개 첫-batch 표본에서 전체 시각·Temporal gradient cosine 모두 양수. 관측 범위에서 두 손실이 반대 방향으로 경쟁한다는 가설 미지지
- client 업데이트 집계 보존 비율은 PEL에서 9 rounds 증가·11 rounds 감소. 로컬 손실 방향과 여러 local steps 후 client 간 업데이트 방향은 다른 측정
- 이벤트별 기준·PEL 진단의 20-round ROC/AP는 각 이전 실행과 정확 일치; 새 실행 기록 회차는 각각 run002로 보존
- 한계: 단일 seed, test에서 최고 round 선택, 두 진단 manifest의 dirty 작업 트리, 데이터 checksum·소스 변경 목록 부재. 성능 하락의 인과 원인은 미확정

장면별 네 조건 중 기존 성능 실험은 완료됐지만, **장면별 기준/PEL의 업데이트·손실 gradient 진단은 아직 미완료**다. 성능 실험과 진단 실험을 구분한다.

## 5. 코드와 실행 환경

| 파일 | 역할 |
| --- | --- |
| `train.py`, `inference.py` | 학습·추론 진입점 |
| `fed/client.py`, `fed/server.py` | 로컬 손실 연결, 기존 집계, 진단·평가 저장 |
| `utils/fixed_pel.py` | 동결된 텍스트 기준 생성, 전경·배경 pooling, 전체 클래스 의미 정렬 |
| `utils/pel_diagnostics.py` | gradient cosine·norm, client delta·보존 비율 |
| `utils/pel_experiments.py` | 후속 계수 schedule·pooling·gradient 적용 모듈 선택 |
| `scripts/run_pel_diagnostics.sh` | 한 조건의 진단 학습 실행 |
| `scripts/prepare_pel_experiments.py` | 후속 실험 명령 생성만 수행; 학습 자동 실행 없음 |
| `scripts/analyze_pel_diagnostics.py` | 기존 로그 검증과 통계 산출 |
| `scripts/record_event_baseline_diagnostics.py`, `scripts/record_event_pel_comparison.py` | 특정 수신 실행의 일회성 기록 생성; 새 회차 범용 도구 아님, 기존 출력이 있으면 재실행 실패하도록 보호 |

Linux 실제 학습 환경은 Python 3.11 계열, PyTorch 2.5.1+cu121, RTX 3090 Ti였다. Python 상세 패치 버전·전체 패키지 고정 목록은 미확보이므로 완전히 고정된 재현 환경으로 간주하지 않는다. 저장소 첫 README의 torch 2.1.0+cu118은 원본 안내이며 최근 실행 환경과 다르다.

새 환경 설치가 필요한 경우의 안내이며 이 인수인계 작업에서 설치·학습하지 않는다:

```bash
conda create -n fed-pel-vad python=3.11
conda activate fed-pel-vad
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python train.py --help
```

PyTorch CUDA 빌드와 호스트 드라이버 호환성을 확인한다. 환경 복원을 이유로 호스트 CUDA·드라이버나 다른 Docker 실험 환경을 일괄 변경하지 않는다. 최초 CLIP 로드에는 모델 캐시 또는 다운로드가 필요하다.

## 6. 받는 방법과 데이터 준비 후 실행 순서

```bash
git clone https://github.com/cvjekim/fed-pel-vad.git
cd fed-pel-vad
git switch --detach handover-2026-10-21
```

인수인계 태그는 고정된 검토본이다. 수정할 때는 해당 태그에서 개인 작업 브랜치를 만든다. 저장소 접근 권한과 인증은 저장소 소유자가 별도로 설정한다.

데이터와 가중치는 후속 전달 범위다. 준비 후 확인할 계약은 다음과 같다.

- 학습 feature: 영상별 NumPy 배열, `T × 512`; CSV의 `path`, `label`과 실제 파일 일치 필요
- 분할 CSV: 기존 이벤트별 13개·장면별 9개, 공통 테스트 CSV와 프레임 정답 필요
- `data/`는 Git에서 무시됨. clone만으로 실제 학습 데이터가 준비되는 것은 아님
- loader가 CSV 경로를 그대로 읽으므로 본인 서버의 배치 위치에 맞는 경로 매핑 확인; 데이터 이동 없이 경로가 저절로 해결된다고 가정하지 않음
- 특징·테스트 순서·GT 길이·라벨 매핑 확인. 다른 client의 데이터를 임의 합치지 않음
- 추론 시 전달할 checkpoint와 모델 옵션을 맞춤. projection 옵션은 완료된 성능 실험에서 사용하지 않음

새 실험은 우선 [사전 설계 양식](../harness/templates/research-experiment.md)에 목적·설계·대조군·고정 조건·측정·판단 한계를 작성하고 설명한다. 이후 해당 한 조건의 명령을 실행한다. 예산·라벨·분할·집계를 임의 바꾸지 않는다.

## 7. 산출물 보존 규칙

- 학습 직후: 기존 `save/<실행 ID>/`의 manifest.json, round_metrics.jsonl, pel_diagnostics.jsonl, summary.json과 터미널 로그 보존
- 원시 진단 전달: `artifacts/experiment-metrics/YYYY-MM-DD_dataset_split_method_aggregation_seedN_roundN_runNNN/` 아래 네 파일 보존; 날짜는 실제 시작일 근거, 회차는 같은 실험 계열의 다음 번호
- 모델 가중치·feature는 결과 JSON 전달과 분리; 실제 분리 작업은 후속 진행
- 정식 실험 기록: `docs/experiments/YYYY-MM-DD_실험종류_runNNN.md`의 고정 JSON을 측정값 원본으로 관리
- 결과 추가·정정 시 [record-experiment-results](../.agents/skills/record-experiment-results/SKILL.md)를 따라 전체 통합표도 재생성
- 지식·수식·구조·해석은 [write-seminar-notes](../.agents/skills/write-seminar-notes/SKILL.md)로 같은 주제의 기존 노트 갱신
- 기존 파일 재학습 결과로 덮어쓰기 금지. 독립 최고 AP와 최고 ROC 시점 AP, 최종 성능 분리

## 8. 다음 담당자가 이어갈 작업

1. 태그의 코드·문서·결과를 읽고 데이터·가중치 후속 전달과 실행 환경 확인
2. 장면별 클라이언트 분할의 기준 모델 진단을 왜 추가하는지 사전 문서 작성: 이벤트별 진단과 달리 한 client에 여러 사건이 있는 조건의 상쇄 기준 확보
3. 같은 장면별 분할에서 PEL 의미 정렬 추가 비교를 별도 사전 설계한 뒤 실행; 손실 경쟁·상쇄·평가 곡선 비교
4. 두 분할 진단이 모이면 기존 계획에 따라 PEL 계수·적용 시점·pooling·모듈 후보 우선순위 재검토. 코드가 준비됐다는 이유로 전체 탐색을 자동 시작하지 않음
5. ConceptNet 의미 사전 보완은 별도 변경 요인으로 비교 설계. 클래스 수·클래스별 개념 수·실제 손실 대조 후보 수를 구분
6. 후보 선정 후 추가 seeds 42·2026의 기준/후보 반복. FedNova·FedPGP 결합은 후속 집계 트랙과 범위를 맞춘 뒤 별도 검증

계수 schedule, 균등 pooling, batch 클래스 기여도 보정, 마지막 Temporal 블록 적용, projection, 명시적 FedNova/FedPGP 연결은 구현 준비 상태다. **실제 영상 성능이 검증된 후보로 표시하지 않는다.**

## 9. 이번 전달본의 변경 사항과 검증

- 이전에 미커밋된 이벤트별 PEL 진단 통계·비교 기록·통합표·세미나 갱신을 함께 포함
- 공식 용어와 실험 전체 이름 작성 지침 및 각 스킬·양식의 연결을 포함
- 인수인계 안내와 프로젝트 첫 진입 링크 추가; 학습·집계 수식 변경 없음
- 기존 실험 원시 파일·측정값 보존. 새 실제 데이터 학습 미실행
- 문서 상대 링크·인코딩·형식, 회차 JSON 및 통합표 동기화 검증. 전체 모델 성능 재측정·Mermaid/수학 실제 렌더·새 Linux 설치는 미검증

검증 재실행:

```bash
python .agents/skills/record-experiment-results/scripts/extract_results.py docs/experiments --check
python .agents/skills/record-experiment-results/scripts/update_summary.py docs/experiments
git diff --check
```
