"""Render a Markdown experiment record to PDF. Reads only; never touches run artifacts.

Headless Chrome does the layout because it is the only engine installed here that has the
Noto Sans CJK KR font wired up through fontconfig. The machine-readable experiment-results
block is an HTML comment in the record; a printed report should show it, so it is lifted
into a labelled appendix rather than dropped.

    python scripts/render_report_pdf.py docs/experiments/<record>.md --output <out>.pdf
"""
import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm; }
body { font-family: "Noto Sans CJK KR","Noto Sans CJK JP",sans-serif; font-size: 10pt;
       line-height: 1.65; color: #1a1a1a; margin: 0; }
h1 { font-size: 18pt; font-weight: 600; line-height: 1.35; margin: 0 0 4mm;
     padding-bottom: 3mm; border-bottom: 2px solid #1a1a1a; }
h2 { font-size: 13pt; font-weight: 600; margin: 9mm 0 3mm; padding-bottom: 1.5mm;
     border-bottom: 0.5px solid #c8c8c8; page-break-after: avoid; }
h3 { font-size: 11pt; font-weight: 600; margin: 6mm 0 2mm; page-break-after: avoid; }
p, li { margin: 0 0 2.2mm; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
/* Let long tables split across pages. Pinning them whole makes Chrome emit a clipped
   fragment that overflows the right margin; repeating the header is the usual fix. */
table { border-collapse: collapse; width: 100%; margin: 3mm 0 5mm; font-size: 8.8pt;
        page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 0.5px solid #d0d0d0; padding: 1.6mm 2.2mm; text-align: left;
         vertical-align: top; }
th { background: #f2f2f0; font-weight: 600; }
code { font-family: "DejaVu Sans Mono",monospace; font-size: 8.6pt; background: #f4f4f2;
       padding: 0.4mm 1mm; border-radius: 2px; }
table code { font-size: 7.3pt; background: none; padding: 0; overflow-wrap: anywhere; }
pre { background: #f7f7f5; border: 0.5px solid #e0e0de; border-radius: 3px;
      padding: 2.5mm 3mm; font-size: 7.6pt; line-height: 1.45; overflow-wrap: anywhere;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
pre.json { font-size: 6.8pt; line-height: 1.4; }
strong { font-weight: 600; }
a { color: #1a1a1a; text-decoration: underline; }
blockquote { margin: 3mm 0; padding-left: 4mm; border-left: 2px solid #d0d0d0; color: #444; }
"""

APPENDIX_HEADING = "부록. 기계 판독용 실험 결과 블록"
APPENDIX_NOTE = "원본 기록에서는 HTML 주석으로 보존된다. 수치 추출 스크립트가 읽는 정본이다."


def build_html(markdown_text, title):
    import markdown as md

    block = re.search(r'<!-- experiment-results\n(.*?)\n-->', markdown_text, re.S)
    body_markdown = markdown_text[:block.start()].rstrip() if block else markdown_text
    body = md.markdown(body_markdown,
                       extensions=['tables', 'fenced_code', 'sane_lists', 'toc'])
    if block:
        body += (f'\n<h2>{APPENDIX_HEADING}</h2>\n<p>{APPENDIX_NOTE}</p>\n'
                 f'<pre class="json">{html.escape(block.group(1))}</pre>')
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head>'
            f'<body>{body}</body></html>')


def find_chrome():
    for name in ('google-chrome', 'chromium', 'chromium-browser'):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit('No Chrome or Chromium on PATH; cannot render PDF')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default=None,
                        help='PDF document title; defaults to the record\'s first heading')
    args = parser.parse_args()

    text = args.record.read_text(encoding='utf-8')
    heading = re.search(r'^#\s+(.+)$', text, re.M)
    title = args.title or (heading.group(1).strip() if heading else args.record.stem)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        page = Path(directory, 'report.html')
        page.write_text(build_html(text, title), encoding='utf-8')
        subprocess.run([find_chrome(), '--headless', '--disable-gpu', '--no-sandbox',
                        '--no-pdf-header-footer',
                        f'--print-to-pdf={args.output.resolve()}', page.as_uri()],
                       check=True, capture_output=True)
    print(f'Wrote {args.output} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
