# 표 작성 규칙

공식 기준: [CVPR author kit](https://github.com/cvpr-org/author-kit), [cvpr.sty](https://github.com/cvpr-org/author-kit/blob/main/cvpr.sty). 확인일 2026-10-07. 공식 스타일은 Times, booktabs와 plain/small caption 설정을 사용한다. 아래 상단 caption·선·강조·정밀도 규칙은 이 프로젝트의 결과표 작성 관례도 포함한다.

- 논문 삽입용 원본은 table/table* float와 caption, label, tabular를 포함한다. 필요한 패키지는 booktabs, graphicx, amsmath, caption이다.
- caption은 위에 두고 데이터셋·분할·단위·평가 조건을 독자가 이해하도록 적는다. UCF AUC와 XD AP는 컬럼 이름에서 구분한다.
- 수평선은 top/mid/bottomrule, 그룹 부분선은 cmidrule을 사용한다. 수직선·셀 격자선은 사용하지 않는다.
- 한 열 3.25in, 양 열 6.875in을 기준으로 읽을 수 있게 배치한다. 과도한 축소보다 헤더 줄바꿈이나 양 열 표를 선택한다.
- 방법 이름은 왼쪽, 수치는 오른쪽 정렬한다. 지표 방향은 수식 화살표로 표시하고 필요하면 caption에서 설명한다.
- mean과 std는 같은 단위로 표현하고 seed 수를 명시한다. 단일 실행에 임의 오차 막대를 붙이지 않는다.
- 비교 조건이 같을 때만 최고를 bold 처리한다. 논문 인용치와 재현 결과는 별도 그룹으로 나누며 SOTA 목표를 달성값으로 쓰지 않는다.
- 입력 텍스트의 LaTeX 특수문자 %, _, &, #, $, {, } 등을 escape한다. 수식은 의도적으로 작성한다.
- 원본 주석에는 출처 파일·페이지/run ID, commit, 실험/보고일 구분, dataset·split·집계·optimizer·개인화·PEL·예산·seed·평가 선택 정책을 남긴다. 모르면 unknown으로 표시한다.
- best AUC와 그 round AP, 독립 best AP, 최종 성능을 서로 다른 헤더로 표시한다.

renderer는 원본 float를 임시 minipage로 바꾸어 Times/caption 스타일로 컴파일한다. 원본 table fragment는 변경하지 않는다. 한글은 XeLaTeX의 kotex를 사용한다. 색·그림으로 표를 다시 그리지 않고 같은 LaTeX를 래스터화한다.
