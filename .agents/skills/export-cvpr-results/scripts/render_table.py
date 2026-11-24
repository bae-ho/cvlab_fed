"""Compile one paper-ready LaTeX table fragment to a cropped PNG/JPG."""
import argparse
from datetime import date
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError("Command failed: " + command[0] + "\n" +
                           (result.stdout + result.stderr)[-7000:])
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex", type=Path)
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width-in", type=float, default=3.25)
    parser.add_argument("--engine", choices=["pdflatex", "xelatex"], default="pdflatex")
    parser.add_argument("--table-number", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.tex.resolve()
    if source.suffix.lower() != ".tex" or not source.is_file():
        parser.error("Input must be an existing .tex file")
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_(.+)", source.stem)
    if not match:
        parser.error("Filename must be YYYY-MM-DD_experiment-and-conditions.tex")
    try:
        date.fromisoformat(match[1])
    except ValueError:
        parser.error("Invalid date in filename")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', source.stem) or source.stem.endswith((" ", ".")):
        parser.error("Filename contains Windows-invalid characters")
    if not 72 <= args.dpi <= 1200 or not 1 <= args.width_in <= 12 or args.table_number < 1:
        parser.error("DPI must be 72..1200, width 1..12 inches, table number >= 1")
    output = source.with_suffix("." + args.format)
    if output.exists() and not args.overwrite:
        parser.error("Output exists; choose distinct conditions or explicitly use --overwrite")
    for command in [args.engine, "pdftoppm"]:
        if shutil.which(command) is None:
            parser.error("Missing tool: " + command)
    try:
        from PIL import Image, ImageChops
    except ImportError:
        parser.error("Missing Pillow: install Pillow in the rendering environment")
    fragment = source.read_text(encoding="utf-8-sig")
    # Ignore full-line comments for matching; retain them in the source artifact.
    content = "\n".join(line for line in fragment.splitlines()
                        if not line.lstrip().startswith("%"))
    pattern = r"\A\s*\\begin\{(table\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}\s*\Z"
    found = re.fullmatch(pattern, content, flags=re.S)
    if not found or re.search(r"\\begin\{table\*?\}", found[2]):
        parser.error("Supply exactly one table/table* fragment, without document preamble")
    body = re.sub(r"\\caption(?=\s*[\[{])", r"\\captionof{table}", found[2])
    if "\\captionof{table}" not in body:
        parser.error("Table must have a caption")
    fonts = (r"\usepackage{fontspec}" + "\n" + r"\setmainfont{TeX Gyre Termes}" + "\n" +
             r"\usepackage{kotex}") if args.engine == "xelatex" else r"\usepackage{times}"
    wrapper = (r"\documentclass[10pt]{article}" + "\n" + fonts + "\n" +
               r"\usepackage{booktabs,amsmath,amssymb,graphicx}" + "\n" +
               r"\usepackage[format=plain,labelformat=simple,labelsep=period,font=small,compatibility=false]{caption}" + "\n" +
               r"\usepackage[active,tightpage]{preview}" + "\n" +
               r"\PreviewEnvironment{minipage}" + "\n" +
               r"\setlength{\PreviewBorder}{4pt}" + "\n" +
               r"\setlength{\textwidth}{" + str(args.width_in) + "in}\n" +
               r"\begin{document}" + "\n" +
               r"\setcounter{table}{" + str(args.table_number - 1) + "}\n" +
               r"\noindent\begin{minipage}{" + str(args.width_in) + "in}\n" + body + "\n" +
               r"\end{minipage}" + "\n" + r"\end{document}" + "\n")
    with tempfile.TemporaryDirectory(prefix="cvpr-table-") as directory:
        work = Path(directory)
        (work / "render.tex").write_text(wrapper, encoding="utf-8")
        log = ""
        for _ in range(2):
            log = run([args.engine, "-no-shell-escape", "-interaction=nonstopmode",
                       "-halt-on-error", "render.tex"], work)
        if "Overfull \\hbox" in log or "Overfull \\vbox" in log:
            raise RuntimeError("Table overflows its width/height. Adjust layout or --width-in.")
        run(["pdftoppm", "-png", "-r", str(args.dpi), "render.pdf", "page"], work)
        pages = list(work.glob("page-*.png"))
        if len(pages) != 1:
            raise RuntimeError(f"Expected one table page; got {len(pages)}")
        with Image.open(pages[0]) as img:
            rgb = img.convert("RGB")
            bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
            if not bbox:
                raise RuntimeError("Rendered image is blank")
            margin = max(8, round(args.dpi / 30))
            box = (max(0, bbox[0]-margin), max(0, bbox[1]-margin),
                   min(rgb.width, bbox[2]+margin), min(rgb.height, bbox[3]+margin))
            cropped = rgb.crop(box)
            settings = {"dpi": (args.dpi, args.dpi)}
            if args.format == "jpg":
                settings.update(quality=95, subsampling=0)
            # Exclusive creation prevents accidental overwrite, including races.
            mode = "wb" if args.overwrite else "xb"
            with output.open(mode) as stream:
                cropped.save(stream, format="JPEG" if args.format == "jpg" else "PNG", **settings)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
        raise SystemExit(str(error))
