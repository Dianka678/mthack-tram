#!/usr/bin/env python3
"""Longest available chronological holdouts for a year-capable hourly baseline."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from backtest_baseline import evaluate, load


FOLDS = [
    ("mar_oct_8mo", date(2025, 2, 28), date(2025, 3, 1), date(2025, 10, 31)),
    ("may_oct_6mo", date(2025, 4, 30), date(2025, 5, 1), date(2025, 10, 31)),
    ("jul_oct_4mo", date(2025, 6, 30), date(2025, 7, 1), date(2025, 10, 31)),
]


def peak_metrics(history, forecast):
    sums = defaultdict(int)
    for (route, _, hour), value in history.items():
        sums[route, hour] += value
    routes = sorted({route for route, _ in sums})
    peak = {route: {hour for _, hour in sorted(
        ((sums[route, h], h) for h in range(24)), reverse=True)[:4]}
        for route in routes}
    err = actual = count = 0
    for route, _, hour, value, pred in forecast:
        if hour in peak[route]:
            err += abs(pred - value)
            actual += value
            count += 1
    return {"hours": count, "actual_sum": actual, "absolute_error_sum": err,
            "wape_score": round(1 - err / actual, 6)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    data = load(args.train) | load(args.test)
    routes = sorted({key[0] for key in data})
    report = {"metric": "1 - absolute_error_sum / actual_sum (missing key is zero only for scoring)",
              "model": "weekday × hour profile with 20% route-hour shrinkage",
              "external_sources": "none, as-of historical weather/traffic/geometry not available in supplied labels",
              "full_year_validated": False,
              "folds": {}}
    for name, cutoff, begin, end in FOLDS:
        history = {key: value for key, value in data.items() if key[1] <= cutoff}
        truth = {key: value for key, value in data.items() if begin <= key[1] <= end}
        report["folds"][name] = {"cutoff": str(cutoff), "from": str(begin),
                                  "to_inclusive": str(end), "models": {}}
        for stat in ("mean", "median"):
            metrics, predictions = evaluate(history, truth, routes,
                                            date(2025, 1, 1), cutoff, begin, end, stat)
            report["folds"][name]["models"][stat] = {
                "metrics": metrics, "peak": peak_metrics(history, predictions)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for name, fold in report["folds"].items():
        print(name, [(stat, obj["metrics"]["all"]["wape_score"])
                     for stat, obj in fold["models"].items()])


if __name__ == "__main__":
    main()
