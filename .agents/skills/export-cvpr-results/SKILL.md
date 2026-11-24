---
name: export-cvpr-results
description: 실험 결과를 CVPR 스타일의 편집 가능한 LaTeX 표 원본으로 저장하고 같은 원본을 컴파일해 날짜·실험명·조건 이름의 PNG 또는 JPG 이미지로 저장한다. 실험 결과표 작성·내보내기 요청에 사용한다.
---

# CVPR 실험 결과표 저장

## 입력과 원본
요청된 결과와 로그·보고표를 읽고 실험 날짜, 이름, 조건, 지표, 단위, seed, checkpoint 선택 기준을 확인한다. [실험 기록 양식](../../../harness/templates/research-experiment.md)과 [두 트랙 경계](../../../docs/research/two-track-scope.md)를 따른다. 표 작성 때문에 집계·학습 설정을 변경하거나 새 학습을 시작하지 않는다.

보고·표·로그가 다르면 출처별 차이를 밝힌다. 최고 ROC round의 AP와 독립 최고 AP, best/final, seed별 값과 평균을 구분한다. 미관측은 -- 또는 N/A로 표시하고 숫자를 만들지 않는다. 비교 가능한 조건만 같은 표에서 강조한다.

## 저장 규칙
사용자가 경로를 지정하면 따른다. 기본은 저장소의 `artifacts/experiment-tables/`이다. 원본 `.tex`와 `.png` 또는 `.jpg`를 같은 폴더와 stem으로 저장한다.

stem은 `YYYY-MM-DD_실험이름및조건`이다. 날짜는 실험 날짜를 사용하며 보고일만 확인되면 report-date라고 원본 주석에 표시한다. 날짜를 모르면 확인하며 현재 날짜를 실험일로 대체하지 않는다. 예: `2026-10-02_cctv_event_pel4vad_round20.png`.

이름에는 dataset, scene/event, 방법·주요 조건을 구별할 만큼 담는다. Windows 금지 문자 `< > : " / \ | ? *`, 제어문자, 끝의 공백·마침표는 피한다. 공백은 underscore로 바꾸되 한글은 보존한다. 여러 표/seed를 구분하려면 조건 또는 표 목적을 이름에 추가한다. 기존 파일은 기본적으로 덮어쓰지 않는다.

## LaTeX 작성
[표 규칙](references/table-style.md)을 읽고 [템플릿](assets/table-template.tex)을 복사해 편집한다. UTF-8 plain text로 저장한다. 논문에 바로 넣을 수 있는 `table` 또는 `table*` fragment를 원본으로 보존하며 document wrapper는 렌더링 스크립트가 임시로 만든다.

상단 caption, label, Times 계열 글꼴, booktabs 수평선을 사용한다. 표의 수직선과 장식적 배경색은 피한다. 공식 author kit에 넣을 때는 해당 연도 cvpr.sty를 사용한다. standalone 이미지는 표 글꼴과 caption 설정을 재현하며 전체 투고 문서 규격을 인증하는 것은 아니다.

원본 주석에 출처, 날짜의 의미, 실제 설정·단위·선택 정책·seed 수·미확인 사항을 기록한다. 원시 정밀도는 주석에 보존하고 표시용 반올림은 일관되게 한다. percent와 percentage point, mean/std를 구분한다. 최고 강조는 표시값의 반올림 동률과 비교 가능성을 고려한다. 한글 caption이 필요하면 스크립트의 `--engine xelatex`를 사용한다.

## 렌더링과 확인
저장소 루트에서 예시:

```powershell
python .agents/skills/export-cvpr-results/scripts/render_table.py artifacts/experiment-tables/2026-10-02_cctv_event_pel4vad_round20.tex --format png --dpi 300
```

JPG는 `--format jpg`를 사용한다. 단일 열 폭은 기본 3.25in, 넓은 표는 `--width-in 6.875`를 사용한다. 필요한 경우 `--table-number`로 번호를 맞춘다. 1개 fragment만 넣는다. 스크립트는 pdflatex 또는 xelatex, pdftoppm, Pillow가 필요하다. 없는 도구는 환경 권한에 따라 준비하고, 불가능하면 원본을 보존해 렌더링 미완료 이유를 보고한다. 생성되지 않은 이미지를 완료로 보고하지 않는다.

실패 시 LaTeX 오류와 overfull box를 확인해 폭/줄바꿈을 수정한다. 수치를 바꾸어 맞추지 않는다. 성공 후 이미지를 실제로 열어 글자·선·수식·caption·잘림과 원본 수치 일치를 확인한다. 파일명과 한글 인코딩도 확인한다. 원본·이미지 링크와 출처·미확인 조건을 완료 보고에 포함한다.

## 공식 용어와 축약 표현 규칙
문서를 생성·정리·갱신할 때 [공통 용어 지침](../../../docs/writing-guidelines.md)을 반드시 따른다. 논문·명세·공식 문서에서 확인한 공식 용어와 공식 약어만 사용한다. 어시스턴트가 자체적으로 만든 축약 표현은 사용하지 않는다. 공식 명칭이 없는 실험·변형은 대상·방법·조건·목적이 이해되도록 풀어서 쓰며, 조사 생략과 단어 나열은 의미가 분명한 경우에만 허용한다.

제목·파일명·표의 행과 열·통합표·세미나 문서·도식 label·수식 설명·완료 보고에도 동일하게 적용한다. 코드 식별자·원본 로그는 정확히 인용하되 독자용 완전한 설명을 병기한다. 기존 기계 판독 필드를 임의 변경하지 않는다. 기존 스킬의 짧은 예시보다 이 규칙이 우선하며 저장 전에 공식성·명확성을 확인한다.
