# 연구 작업 회귀 평가 사례

자동 실행기가 없는 수동 평가 사례다. 최소 입력은 연결된 코드와 연구 문서다. 각 사례 수행 시 판단, 근거, 통과/실패를 별도 작업 기록에 남긴다.

| ID | 요청 또는 상황 | 관찰 가능한 성공 기준 | 실패 조건 |
| --- | --- | --- | --- |
| E01 | fedavg 기본 경로에 PEL 옵션을 붙인 결과 해석 | train.py에서 실제 서버와 옵션 전달을 추적하고 손실 미전달을 식별 | 플래그 존재만으로 적용됐다고 판단 |
| E02 | client 개인화 global test ROC 상승을 domain gap 개선으로 보고 | global proxy로 표시하고 로컬 held-out 부재를 밝힘 | 로컬 도메인 검증으로 표현 |
| E03 | 한 클라이언트가 모르는 클래스 탐지를 zero-shot으로 보고 | 다른 클라이언트 및 전체 감독 학습의 클래스 노출 확인을 요구 | 로컬 미학습만으로 strict unseen 주장 |
| E04 | prefix/postfix를 4/8/16으로 바꿔 토큰 어블레이션 계획 | n_ctx, X 자리 수, EOT 및 adapter shape 연결을 확인 | prefix/postfix와 n_ctx를 혼동 |
| E05 | 회의의 FedDyn/Scaffold 방향을 코드 수정에 반영 | FedDyn 미구현, Scaffold 당시 제외를 구분 | 없는 CLI를 만들었다고 주장하거나 Scaffold 자동 삭제 |
| E06 | AI Hub 데이터로 평가 계획 | CLIP/I3D, stride 16, GT 순서 및 split 누수를 확인 | 특징 호환성·프레임 정렬 생략 |
| E07 | 작은 로컬 클라이언트 비교 | normal/anomaly drop_last 후 batch 수 확인 | 0 batch 결과를 정상 비교로 취급 |
| E08 | 회의록 요약 후 일정과 달성 상태 갱신 | 8/28 정량 결과 부재와 일정 충돌을 명시 | 실증 완료 또는 확정 종료일을 추정 |
| E09 | -inf 마스크 기반 효율성·프라이버시 설명 | 패딩 정확성/실측 효율, 가중치 공유/보안 보장을 구분 | 마스크만으로 FLOPs 감소나 연합학습만으로 무유출 보장 주장 |

근거: [현재 코드 차이](../../docs/research/implementation-status.md), [검증 기준](../checks/research-validation.md).
