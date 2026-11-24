"""Rebuild a date-sorted Markdown comparison table from all experiment records."""
import argparse
from collections import OrderedDict
import html
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

from extract_results import collect, load_record

MARKER = "<!-- experiment-results-summary: generated v1 -->"
KEY_FIELDS = ["document", "date", "date_kind", "experiment_type", "attempt", "track", "status",
              "method", "dataset", "split", "seed", "round", "statistic", "selection",
              "provenance", "conditions_json"]


def cell(value):
    if value is None:
        return "--"
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\r", " ").replace("\n", " ").replace("`", "&#96;")


def link(document, output):
    relative = os.path.relpath(Path(document).resolve(), output.parent.resolve()).replace("\\", "/")
    return "[record](" + quote(relative, safe="/.-_") + ")"


def build_summary(root, output):
    _, rows = collect(root, allow_empty=True)
    records = []
    for path in sorted(root.rglob("*.md")):
        if path.name.lower() not in {"readme.md", "results-summary.md"}:
            data, _ = load_record(path)
            records.append((path, data))
    rows.sort(key=lambda row: (row["date"], row["experiment_type"], row["attempt"],
                              row["method"], row["dataset"], row["split"],
                              -1 if row["seed"] is None else row["seed"],
                              -1 if row["round"] is None else row["round"],
                              row["statistic"], row["selection"], row["conditions_json"],
                              row["provenance"]))
    metrics = sorted({(row["metric"], row["unit"]) for row in rows})
    groups = OrderedDict()
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        entry = groups.setdefault(key, {"row": row, "values": {}, "sources": []})
        metric = (row["metric"], row["unit"])
        if metric in entry["values"]:
            raise ValueError("Duplicate metric in comparison row")
        entry["values"][metric] = row["value"]
        if row["source"] and row["source"] not in entry["sources"]:
            entry["sources"].append(row["source"])
    lines = [MARKER, "# Experiment results: chronological comparison", "",
             "Generated from per-run experiment-results JSON. Update the records, then regenerate this file.",
             "Values retain source precision and units. `--` means unavailable, not zero. Dates marked report are report dates.",
             "Different conditions, units, statistics and selection policies are kept separate; this table does not certify comparability.", ""]
    if not groups:
        lines += ["No measurements recorded yet.", ""]
    else:
        headers = ["Date", "Date kind", "Type", "Attempt", "Track", "Status", "Method", "Dataset", "Split",
                   "Seed", "Round", "Statistic", "Selection", "Provenance", "Conditions"]
        headers += [f"{metric} ({unit})" for metric, unit in metrics] + ["Source", "Record"]
        lines += ["| " + " | ".join(cell(h) for h in headers) + " |",
                  "| " + " | ".join("---" for _ in headers) + " |"]
        for entry in groups.values():
            row = entry["row"]
            values = [row[field] for field in KEY_FIELDS if field != "document"]
            values[3] = f"run{row['attempt']:03d}"
            values += [entry["values"].get(metric) for metric in metrics]
            values += ["; ".join(entry["sources"])]
            lines.append("| " + " | ".join(cell(value) for value in values) + " | " + link(row["document"], output) + " |")
        lines.append("")
    empty = [(path, data) for path, data in records if not data["measurements"]]
    if empty:
        lines += ["## Runs without measurements", "", "| Date | Date kind | Type | Attempt | Status | Record |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for path, data in sorted(empty, key=lambda item: (item[1]["date"], item[1]["experiment_type"], item[1]["attempt"])):
            values = [data["date"], data["date_kind"], data["experiment_type"], f"run{data['attempt']:03d}", data["status"]]
            lines.append("| " + " | ".join(cell(value) for value in values) + " | " + link(str(path), output) + " |")
        lines.append("")
    return "\n".join(lines)


def update_summary(root, output):
    if not root.is_dir():
        raise ValueError("Input must be a directory of experiment records")
    if output.name.lower() != "results-summary.md":
        raise ValueError("Summary must be named results-summary.md so extraction skips it")
    # Validate the complete source before any write. Preserve unrelated content.
    text = build_summary(root, output)
    if output.exists() and not output.read_text(encoding="utf-8-sig").startswith(MARKER):
        raise ValueError("Existing summary is not generator-owned; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output.parent, prefix=".summary-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(update_summary(args.input, args.output or args.input / "results-summary.md"))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise SystemExit(str(error))
