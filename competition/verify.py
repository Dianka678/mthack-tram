"""Verify the competition artifact against the organizer's sample grid.

Usage: python3 competition/verify.py --sample data/test_submission.csv \
       --submission competition/submission.csv --model competition/model.json
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != ["route", "date", "hour", "prediction"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    sample, submission = read_csv(args.sample), read_csv(args.submission)
    dates = [date(2025, 11, 1) + timedelta(days=offset) for offset in range(61)]
    expected = {(route, day.isoformat(), str(hour)) for route in (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
                for day in dates for hour in range(24)}
    keys = [(int(row["route"]), row["date"], row["hour"]) for row in submission]
    assert len(sample) == len(submission) == len(expected) == 14640
    assert len(set(keys)) == len(keys) and set(keys) == expected
    assert all((s["route"], s["date"], s["hour"]) == (o["route"], o["date"], o["hour"])
               for s, o in zip(sample, submission))
    for row in submission:
        value = row["prediction"]
        if not value.isascii() or not value.isdecimal() or int(value) < 0:
            raise ValueError(f"Invalid prediction: {value!r}")
    assert Counter(int(row["route"]) for row in submission) == {route: 1464 for route in (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)}
    model = json.loads(args.model.read_text(encoding="utf-8"))
    assert model["as_of"] == "2025-10-31" and model["cold_start_route"] == 5
    assert model["source_sha256"] == {key: manifest["input_sha256"][key] for key in ("train", "test")}
    assert sha256(args.sample) == manifest["input_sha256"]["sample"]
    assert sha256(args.submission) == manifest["artifact_sha256"]["submission.csv"]
    assert sha256(args.model) == manifest["artifact_sha256"]["model.json"]
    print("OK: 14,640 rows, exact sample order and grid, integral nonnegative predictions, SHA matches")


if __name__ == "__main__":
    main()
