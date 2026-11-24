"""Validate Markdown experiment records and extract long-form CSV (stdlib only)."""
import argparse
import csv
from datetime import date
import io
import json
import math
from pathlib import Path
import re
import sys

FIELDS = ["document", "date", "date_kind", "experiment_type", "attempt", "track", "status",
          "method", "dataset", "split", "metric", "value", "unit", "seed", "round",
          "statistic", "selection", "provenance", "source", "conditions_json"]
BLOCK = re.compile(r"^```experiment-results[ \t]*\r?\n(.*?)^```[ \t]*\r?$", re.M | re.S)
UNITS = {"ratio", "percent", "percentage_point", "count", "ms", "seconds", "bytes", "unitless"}
STATS = {"single", "mean", "std", "delta", "n"}
STATUSES = {"planned", "running", "completed", "failed", "interrupted", "reported"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def load_record(path):
    text = path.read_text(encoding="utf-8-sig")
    blocks = BLOCK.findall(text)
    require(len(blocks) == 1, "Expected exactly one experiment-results block")
    data = json.loads(blocks[0], object_pairs_hook=strict_object, parse_constant=reject_constant)
    require(isinstance(data, dict), "Record must be an object")
    required = {"schema_version", "date", "date_kind", "experiment_type", "attempt", "track",
                "status", "conditions", "measurements"}
    require(required <= data.keys(), "Missing document fields: " + str(sorted(required-data.keys())))
    require(type(data["schema_version"]) is int and data["schema_version"] == 1, "Unsupported schema")
    require(isinstance(data["date"], str), "date must be a string")
    date.fromisoformat(data["date"])
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["date"]), "date must be YYYY-MM-DD")
    kind = data["experiment_type"]
    require(isinstance(kind, str) and bool(kind) and not re.search(r'[<>:"/\\|?*\s\x00-\x1f]', kind)
            and not kind.endswith("."), "Invalid experiment_type")
    require(type(data["attempt"]) is int and data["attempt"] > 0, "attempt must be positive integer")
    require(path.stem == f'{data["date"]}_{kind}_run{data["attempt"]:03d}', "Filename and record disagree")
    require(data["date_kind"] in {"experiment", "report"}, "Invalid date_kind")
    require(isinstance(data["track"], str) and bool(data["track"].strip()), "Missing track")
    require(data["status"] in STATUSES, "Invalid status")
    require(isinstance(data["conditions"], dict), "conditions must be an object")
    require(isinstance(data["measurements"], list), "measurements must be an array")
    rows, seen = [], set()
    needed = {"method", "dataset", "split", "metric", "value", "unit", "seed", "round", "statistic",
              "selection", "provenance", "source", "conditions"}
    for index, item in enumerate(data["measurements"], 1):
        require(isinstance(item, dict) and needed <= item.keys(), f"Measurement {index}: missing fields")
        for field in ["method", "dataset", "split", "metric", "selection"]:
            require(isinstance(item[field], str) and bool(item[field].strip()), f"Invalid {field}")
        require(item["unit"] in UNITS and item["statistic"] in STATS, "Invalid unit/statistic")
        require(item["provenance"] in {"observed", "reported", "derived"}, "Invalid provenance")
        require(item["seed"] is None or type(item["seed"]) is int, "seed must be integer/null")
        require(item["round"] is None or (type(item["round"]) is int and item["round"] >= 0),
                "round must be nonnegative integer/null")
        require(isinstance(item["source"], str), "source must be a string")
        require(isinstance(item["conditions"], dict), "measurement conditions must be an object")
        value = item["value"]
        require(value is None or (type(value) in (int, float) and math.isfinite(value)),
                "value must be finite numeric/null")
        if value is not None:
            require(bool(item["source"].strip()), "Numeric value needs source")
            if item["statistic"] in {"single", "mean"} and item["unit"] in {"ratio", "percent"}:
                require(0 <= value <= (1 if item["unit"] == "ratio" else 100), "Metric out of range")
            if item["statistic"] in {"std", "n"}:
                require(value >= 0, "std/n must be nonnegative")
            if item["statistic"] == "n":
                require(item["unit"] == "count" and float(value).is_integer(), "n must be integer count")
        conditions = {**data["conditions"], **item["conditions"]}
        key = tuple(item[f] for f in ["method", "dataset", "split", "metric", "unit", "seed", "round",
                                      "statistic", "selection"]) + (json.dumps(conditions, sort_keys=True),)
        require(key not in seen, "Duplicate measurement identity")
        seen.add(key)
        row = {field: data[field] for field in ["date", "date_kind", "experiment_type", "attempt",
                                               "track", "status"]}
        row.update({field: item[field] for field in needed - {"conditions"}})
        row["document"] = path.as_posix()
        row["conditions_json"] = json.dumps(conditions, ensure_ascii=False, sort_keys=True)
        rows.append(row)
    return data, rows


def collect(root, kind=None, allow_empty=False):
    require(root.exists(), "Input path does not exist")
    files = [root] if root.is_file() else sorted(root.rglob("*.md"))
    records, rows, seen = 0, [], set()
    for path in files:
        if path.name.lower() in {"readme.md", "results-summary.md"}:
            continue
        try:
            data, values = load_record(path)
            key = (data["experiment_type"], data["attempt"])
            require(key not in seen, "Duplicate experiment_type/attempt across dates")
            seen.add(key)
            if kind is None or data["experiment_type"] == kind:
                records += 1
                rows.extend(values)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"{path}: {error}") from error
    require(records > 0 or allow_empty, "No matching experiment records")
    return records, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--type", dest="kind")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("--check does not produce a CSV")
    records, rows = collect(args.input, args.kind)
    if args.check:
        print(f"Validated {records} records, {len(rows)} measurements")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8-sig", newline="") as stream:
            stream.write(buffer.getvalue())
        print(f"Exported {len(rows)} measurements to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(buffer.getvalue())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, OSError) as error:
        raise SystemExit(str(error))
