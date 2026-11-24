# PEL 표현 정렬과 연합학습 호환성 진단·개선 계획

- 날짜: 2026-10-08
- 상태: 1–5단계의 진단·후보·명령 준비·취합·후속 연결 코드 구현, CPU 합성 검증 완료; 새 실제 데이터 학습 미실행
- 트랙: 김배호 연구원의 PEL 적용 지원
- 근거: 사용자 요청과 4개 UCF 실행의 보고 로그
- 적용 절차: [plan-federated-ablation](../../../.agents/skills/plan-federated-ablation/SKILL.md), [실험 기록 템플릿](../../../harness/templates/research-experiment.md)

## 요청과 성공 기준

event와 scene에서 PEL 효과가 다른 원인을 진단하고, 기본 FedAvg를 유지하면서 PEL 손실·적용 위치를 개선한다. 이후 집계 트랙과 결합 가능성을 평가한다. 최종 성능 목표의 원본은 [docs/README.md](../../README.md)이며, 현재 결과로 SOTA 달성이나 미학습 이벤트 검출을 주장하지 않는다.

- 동작 성공: 의도한 시각 모듈로 유한 gradient 전달, 텍스트 기준 고정, 옵션 off 경로 및 기존 집계 보존
- 성능 성공: 동일 분할·예산·seed·선택 정책에서 기준 대비 UCF 주 지표 ROC-AUC 개선; AP·최종 성능·후기 하락도 함께 보고
- 반복성: 선정 후보와 기준의 paired seed 비교 및 평균·표준편차 기록
- 원인 주장: 실제 손실 gradient·업데이트·클래스 분포 측정으로 뒷받침; 점수 차이만으로 평균 희석이나 과적합을 확정하지 않음

## 관찰과 현재 구현

[결과 원본](../../research/2026-10-08-ucf-event-scene-results.md), [실험 기록](../../experiments/README.md), [수치 JSON](../../../artifacts/experiment-metrics/2026-10-08_ucf_event_scene_reported_rounds.json).

| 기준 대비 PEL 변화 (%p) | event | scene |
| --- | ---: | ---: |
| 최고 ROC | +0.92 | -0.17 |
| 독립 최고 AP | +1.02 | +0.32 |
| 최고 ROC 시점 AP | +2.14 | -0.18 |
| 마지막 ROC | -1.24 | -1.52 |
| 마지막 AP | -1.36 | -0.80 |

- 기존 결과: seed 88888888 한 개, 사용자 제공 로그 기준; 최고값은 공통 test에서 선택
- event PEL 최고 ROC round 11, scene PEL round 8; 두 분할 모두 마지막 점수 하락
- 이번 후보: 원본 PEL4VAD 전체 재현이 아닌 fixed-bank 변형
- 텍스트 기준: UCF 클래스 이름 14개; 설명형 프롬프트 사전 미전달
- PEL gradient: Temporal 출력 H를 통해 Temporal·위치 임베딩으로 전달; pooling 점수와 고정 텍스트는 detach
- MIL 기반 분류 손실: 동적 프롬프트 분류 경로를 계속 학습
- 별도 학습 가능한 PEL 파라미터 없음; 손실 계수 1.0은 집계하는 모델 파라미터가 아님
- event도 전체 텍스트 후보 사용; 다른 이벤트의 실제 영상 양성까지 로컬에서 학습하는 것은 아님

## 가설과 반증 기준

| ID | 가설 | 필요한 관측 | 해석 제한 |
| --- | --- | --- | --- |
| H1 | 공통 고정 텍스트 기준이 로컬 업데이트의 의미 방향을 정렬 | PEL off/on의 동일 공유 모듈 업데이트 방향 및 집계 후 평가 비교 | 방향 일치 자체가 성능 향상을 보장하지 않음 |
| H2 | MIL 기반 분류 손실과 PEL이 공유 시각 모듈에서 경쟁 | 같은 batch의 두 손실 gradient cosine·norm, round·client·모듈별 분포 | 음의 cosine 몇 건으로 전체 원인을 확정하지 않음 |
| H3 | 점수 기반 전경 선택 오류가 후기 하락에 기여 | pooling 집중도·전경/배경 유사도 추이, pooling 대조군 성능 | 영상 라벨만으로 전경 위치 정답을 확정하지 않음 |
| H4 | scene의 클래스·배경 편향이 효과를 제한 | 실제 사용 클래스 횟수, 클래스별 손실·평가, client별 steps | 여러 이벤트가 있다는 사실 자체를 원인으로 취급하지 않음 |
| H5 | 로컬 전문화·업데이트 상쇄가 글로벌 성능을 제한 | 로컬 학습 전후·집계 후 동일 probe 평가, 업데이트 norm·cosine·보존 비율 | 로컬 held-out 없이 과적합 확정 금지; 공통 test proxy로 명시 |

“PEL 가중치가 평균 후 남는다”는 기존 추측은 H1/H5로 검증한다. scene에서도 최고 AP는 증가했으므로 모든 지표에서 실패했다고 기록하지 않는다. 연합학습과 PEL의 필연적 충돌을 전제하지 않는다.

## 범위와 통제 조건

| 항목 | 고정 조건 |
| --- | --- |
| 데이터 소유 | 기존 UCF event 13 clients / scene 9 clients; 다른 클라이언트 데이터 합침·재분할 없음 |
| 집계 | 기존 FedAvg, 기존 DataLoader 배치 수 기반 계수 및 파라미터 반환·배포 유지 |
| 서버·개인화 | server_optimizer none, personalization_type none, local_param_policy none |
| 예산 | 20 rounds × 10 local epochs, batch 64 per normal/anomaly loader, LR 0.0001, AdamW |
| 실제 steps | 각 client의 min(normal loader, anomaly loader) × local epochs 기록; epoch 동일을 step 동일로 오인하지 않음 |
| 의미 기준 | 동일 클래스 순서·클래스 이름·고정 bank checksum; temperature 0.09 |
| 모델·추론 | 기존 MIL 기반 분류 손실·동적 프롬프트·1-P(Normal), feature·GT·padding 정책 유지 |
| 초기 비교 seed | 88888888; 이후 기준과 선정 후보에 42, 2026 추가, 총 3개 paired seeds |
| 평가 | 기존 test 기반 곡선은 탐색·재현용; 독립 성능 주장에는 동일 validation 선택 정책과 최종 test 평가 필요 |

환경·코드 commit·데이터 manifest는 새 실행 전에 기록한다. 호스트 CUDA/드라이버 변경은 계획에 포함하지 않는다. 학습 실행은 별도 단계이며 이번 문서화에서 수행하지 않는다.

## 순차 실행 계획

### 단계 1 — 진단 구현과 기존 후보 측정

1. `fed/client.py`에 선택적 진단 경로 추가: 동일 forward의 MIL 기반 분류 손실·PEL을 분리해 공유 시각 파라미터 gradient 측정. `autograd.grad` 등으로 측정하고 optimizer의 실제 gradient·step·RNG·batch 순서를 바꾸지 않음.
2. `fed/server.py`에 읽기 전용 업데이트 진단 추가: round 시작 공유 파라미터 대비 각 client delta, 실제 집계 계수와 평균 delta 기록. 집계 함수·계수는 수정하지 않음.
3. 고정 bank·label map·모듈 이름·진단 batch 식별 규칙·측정 빈도·진단 시간 기록. 우선 rounds 1/5/10/15/20의 client별 첫 유효 batch 측정; 업데이트 진단은 매 round.
4. CPU 합성 검증 후 event 기준 → event 현재 PEL → scene 기준 → scene 현재 PEL 순으로 seed 88888888 진단 실행. 기존 로그에는 gradient가 없어 현재 기록으로 대체할 수 없음.

진단 측정:

- 손실 cosine: g_MIL과 g_PEL의 방향 일치도; PEL 적용 전·계수 적용 후 norm 비율 구분
- client delta cosine: 동일 공유 모듈의 업데이트 방향; 모듈별 값과 실제 계수로 가중한 값 함께 기록
- 집계 보존 비율: ||sum_i a_i delta_i|| / sum_i a_i ||delta_i||; 0 분모는 결측 처리. 작을수록 상쇄 가능성, 유용한 특징 소실을 직접 증명하지는 않음
- pooling entropy·전경/배경 텍스트 cosine·클래스별 PEL 기여·실제 클래스 사용 횟수
- round별 ROC/AP, 최종 성능, 진단 실행 overhead; 로컬 모델 평가는 공통 test proxy로 표시하고 선택용 추가 정보로 사용하지 않음

### 단계 2 — PEL 강도·시점 비교

| ID | 후보 | 변경 요인 | 실행 준비 |
| --- | --- | --- | --- |
| B0 | MIL 기반 분류 손실, PEL off | 기준 | 기존 CLI 사용 가능 |
| P0 | 현재 fixed-bank, weight 1.0 | 기존 후보 | 기존 CLI 사용 가능 |
| P1 | fixed-bank, weight 0.1 | 손실 강도 | 기존 CLI 사용 가능 |
| P2 | fixed-bank, weight 0.3 | 손실 강도 | 기존 CLI 사용 가능 |
| P3 | 선정 weight; rounds 1–3 PEL 0, rounds 4–6 선형 증가, 이후 유지 | 활성화 시점 | round 전달 및 계수 schedule 구현 필요 |

P1 event → P1 scene → P2 event → P2 scene 순으로 실행. 고정 강도 후보를 비교한 뒤 P3를 event → scene 순으로 실행한다. 각 실행은 동일 seed 88888888이며 P0/B0는 단계 1 결과를 재사용한다. 추가 임의 탐색은 하지 않는다. 두 분할을 함께 보는 후보 선정은 탐색임을 명시하고 독립 평가로 구분한다.

### 단계 3 — 원인에 따른 적용 방식 비교

아래 후보는 동시에 결합하지 않고 단계 2 선정 조건을 공통 출발점으로 하나씩 비교한다. 진단 근거가 약하면 후보의 우선순위를 다시 기록한다.

| 순서 | 후보 | 목적·선행 작업 |
| --- | --- | --- |
| 1 | score pooling 대 uniform-valid pooling 진단 대조군 | 점수 기반 선택 의존성 확인; uniform은 전경/배경 분리를 약화하는 원인 진단용이며 개선을 보장하지 않음 |
| 2 | PEL 항 안에서만 클래스별 기여 정규화 | 희소 이벤트 지배 여부에 따라 설계; MIL 기반 분류 손실·sampler·집계는 유지. 실제 사용 빈도 기준·상한·결측 클래스 처리 기록 |
| 3 | PEL gradient를 마지막 Temporal 블록으로 제한 | MIL 기반 분류 손실은 기존 전체 시각 경로 학습 유지; gradient 라우팅 구현·경계 검증 필요 |
| 4 | H 뒤 projection head를 둔 PEL | 의미 정렬 완충 가능성 검증; H로 gradient 전달 유지, head-only 학습 금지. 새 key·shape 및 공유·추론 정책을 설계한 뒤 별도 승인 범위로 구현 |

confidence 기반 전경 선택과 설명형 프롬프트 사전은 추가 후보로 보류한다. 진단 없이 새 threshold·사전·모듈을 한 번에 추가하지 않는다. projection은 현재 공유/로컬 정책에 영향을 줄 수 있어, 기존 모듈의 정책을 임의 변경하지 않고 통합 영향과 필요한 범위를 먼저 기록한다.

### 단계 4 — 반복성 및 후속 집계 결합

1. 최종 PEL 후보와 B0를 event/scene 각각 seeds 42, 2026으로 추가 실행; 88888888을 합쳐 3개 seed별 delta·평균·표준편차 보고.
2. round를 반복 표본으로 취급하지 않음. 최고 ROC·해당 AP·독립 최고 AP·최종 값·전체 곡선을 모두 보존.
3. 집계 트랙과 후속 범위가 확정되면 FedAvg/FedNova/FedPGP 각각 PEL off/on의 교차 대조군 구성. 이번 계획은 실행 가능성 검토이며 집계 코드 변경·통합 학습을 시작하지 않음.
4. FedNova는 실제 local steps 및 정규화 규약 확인; PEL 손실 충돌 해결을 자동 기대하지 않음. FedPGP는 개인화·adapter·공유/로컬 상태·global/local 평가를 기록하고 집계만의 효과로 해석하지 않음.
5. 최초 fixed_bank 옵션은 plain FedAvg만 허용했음. 2026-10-08 추가 사용자 요청으로 후속 FedNova/FedPGP 연결 코드를 준비; 명시적인 `--pel_allow_aggregation_integration` 없이 기존 제한 유지. 집계 식·가중치·기존 공유/로컬 정책 변경 없음. 실제 통합 성능 미검증.

## 관련 코드와 근거

- [고정 bank·PEL 손실](../../../utils/fixed_pel.py), [모델](../../../utils/model.py), [클라이언트](../../../fed/client.py), [서버](../../../fed/server.py), [실행 분기](../../../train.py)
- [이전 구현 계획](2026-10-06-fedavg-fixed-anchor-pel.md), [두 트랙 범위](../../research/two-track-scope.md), [검증 기준](../../../harness/checks/research-validation.md)
- [PEL4VAD 원논문](https://arxiv.org/abs/2306.14451): 지식 프롬프트를 통한 이상 하위 클래스 의미 구분; 이번 fixed-bank 변형과 구분
- [MOON 원논문](https://arxiv.org/abs/2103.16257): 연합학습과 대조학습의 결합 가능성 근거; 현재 PEL의 성공 보장 아님
- [FedNova 원논문](https://arxiv.org/abs/2007.07481): 이질적 local update 횟수의 목적 불일치; 모든 표현 충돌을 해결하는 방법 아님

## 실행 단계와 검증 기록

- 2026-10-14 PEL 결과 수신: UCF-Crime 이벤트별 클라이언트의 PEL 추가 진단 완료. [실험 기록](../../experiments/2026-10-14_ucf_event_fixed_bank_pel_run002.md). 전체 시각 gradient 표본 65개 모두 양의 cosine; 로컬 손실 경쟁 가설 미지지. 집계 보존 비율 변화는 round별 혼재. 장면별 두 진단 미완료.

- 2026-10-14 결과 수신: 단계 1의 event 기준 진단 완료. [run002 결과](../../experiments/2026-10-13_ucf_event_baseline_run002.md), [수식·관찰 노트](../../seminar/2026-10-14_손실경쟁과_업데이트상쇄_진단.md). event PEL 및 scene 두 진단은 미완료; 단계 1 전체 완료로 처리하지 않음.

- [x] 코드·기존 결과·관찰과 가설 구분 확인
- [x] 범위·통제 조건·진단·후보 순서 문서화
- [x] 진단 구현 및 옵션 off 회귀·gradient 유한성·집계 보존 CPU 검증
- [ ] 단계 1–2 실행 및 결과 기록
- [x] 단계 2–3 후보 및 단계 4–5 명령 준비·결과 취합·opt-in 연결 구현, CPU 합성 검증
- [ ] 단계 3 후보별 실제 실행
- [ ] 선정 후보 반복성 검증
- [ ] 후속 집계 트랙과 통합 범위 확정 및 교차 비교

| 확인 | 결과 | 근거 |
| --- | --- | --- |
| 현재 gradient·bank·집계·CLI 경계 | 코드 정적 확인 | 위 관련 코드 |
| 기존 4개 실험 | 보고 로그 80개 round 평가, 단일 seed | 결과 문서·수치 JSON |
| 새 후보 학습·원인 검증 | 미실행 | 본 계획의 후속 단계 |

### 2026-10-08 진단 구현 기록

- 활성화: 기존 plain FedAvg 명령에 `--pel_diagnostics` 추가; 기본은 off. UCF/CCTV에서 PEL off 또는 fixed_bank만 허용. 기존 CLI의 집계·개인화 제한을 유지.
- 저장: 기존 실행 출력 디렉토리의 `pel_diagnostics.jsonl`. rounds 1/5/10/15/20의 각 client 첫 epoch 첫 batch에서 손실 gradient·pooling·batch 클래스 수 측정; 매 round의 모든 공유 모듈 delta 진단.
- gradient: cosine, MIL 기반 분류 손실/PEL norm, 계수 적용 전·후 비율, 실제 파라미터 이름. `autograd.grad(retain_graph=True)` 사용; `.grad`와 optimizer 상태를 쓰지 않고 추가 forward·무작위 추출 없음.
- 업데이트: 공통 round 시작 파라미터 대비 client delta와 실제 집계 delta, loader 배치 수 계수, pairwise cosine·계수 곱, client 대 aggregate cosine, 집계 보존 비율 및 부동소수점 잔차. 가중 pair 평균은 유효 cosine의 계수 곱을 정규화해 기록; 0 norm pair 제외.
- pooling: 전경/배경 순서의 entropy·최대 weight, 전경 목표·이상 배경 Normal cosine, class index별 KL 항 횟수·평균. 시각 특징 원본을 파일로 저장하지 않음.
- 다른 진단 정보: 전체 사용 클래스·실제 steps·평균 손실·ROC/AP는 기존 `round_metrics.jsonl`과 round로 연결; seed·코드 commit은 기존 manifest, bank·클래스 순서는 진단 기록에서 확인.
- 비용: `forward_and_diagnostic_seconds`는 해당 forward와 gradient 측정 합산 시간이며 순수 추가 overhead 아님. `update_diagnostic_seconds`는 업데이트 측정 시간. CPU로 이동한 double delta의 추가 RAM·통신 동기화 비용은 실제 환경에서 미측정.
- 회귀 검증: CPU 합성 입력에서 PEL off/on 각각 진단 옵션 on/off의 파라미터·gradient·AdamW 상태·RNG 정확 일치, 두 client 실제 배치 수 가중 평균 정확 일치. 0 norm은 cosine/비율 null; JSON NaN 저장 금지.
- 실제 모델 경로: mock CLIP을 사용한 CPU Temporal·위치 임베딩에서 진단 후 정상 backward와 유한 gradient 확인. pretrained CLIP 다운로드·GPU 사용 없음.
- 자동 추가 로컬 test 평가는 구현하지 않음. 기존 글로벌 평가를 유지하며, 로컬 과적합/전경 위치 정답은 이번 진단만으로 확정할 수 없음. 필요 시 H5의 별도 probe 평가 설계로 진행.
- 검증 명령: `python -m unittest discover -s tests -v`; 실제 데이터 학습은 사용자 요청에 따라 시작하지 않음.

결과 기록은 [record-experiment-results](../../../.agents/skills/record-experiment-results/SKILL.md), 표 출력은 [export-cvpr-results](../../../.agents/skills/export-cvpr-results/SKILL.md)를 사용한다. 실행 전 각 조건의 정확한 명령·코드 commit·manifest·checkpoint 선택 정책을 실험 기록 템플릿에 채운다.

### 후속 후보 코드 준비 기록

- 근거: 2026-10-08 사용자의 5단계 기록 및 남은 코드 사전 작성 요청; 직전 학습 금지 지시 유지
- `utils/pel_experiments.py`: round별 계수 schedule, fixed-loss 선택, base gradient 보존 및 마지막 블록 PEL gradient 라우팅
- `utils/fixed_pel.py`: score/uniform pooling, batch target row별 역빈도 기여 정규화. 초기 계획의 데이터셋/누적 빈도 대신 실제 현재 batch 관측 빈도 사용; absent 클래스는 양성 기여 없음, 전체 텍스트 denominator는 유지
- `utils/model.py`: identity·RNG 보존 projection head, 새 공유 state key `pel_projection.weight`, 기존 추론 logits에서는 미사용
- 기존 서버·client 연결: 새 옵션은 default off; FedNova/FedPGP의 원래 집계·steps·adapter 보존 경로에 fixed-bank 손실만 선택적으로 전달
- `scripts/prepare_pel_experiments.py`: 강도·시점·pooling·모듈·추가 seeds·집계별 off/on 명령의 JSON 생성; 학습 실행 기능 없음
- `scripts/summarize_pel_runs.py`: 기존 파일만 읽어 paired seed delta·평균·표본 표준편차 취합; incomplete/duplicate rounds 및 duplicate seeds 오류 처리. 조건·commit/torch/server 정보로 그룹 구분; 데이터 checksum 검증은 별도 필요
- projection 추론: 해당 후보 checkpoint는 `inference.py --pel_projection`으로 동일 state 구조 구성; head off 기존 checkpoint 형태 유지
- 실험 조건 선택: 추가 옵션은 독립 비교용. weight 0.3, warmup 3/ramp 3 등 준비값은 성능으로 확정된 선정 결과가 아님
- 세미나 기록: [5단계 실험 구현 핵심](../../seminar/2026-10-08_PEL_연합학습_5단계_실험구현.md)
- 검증: CPU 테스트 20개 통과, 9개 Python 소스 문법·문서 링크/인코딩·diff 형식 확인, `fed/aggregation.py` 변경 없음
- 미실행: 실제 영상 학습, Linux RTX3090 실행, 추가 seed 반복, 성능 선정, 집계 결합 성능; Mermaid·수식 실제 렌더 미검증
