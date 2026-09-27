#!/usr/bin/env python3
"""Daily / weekly operational backtest with strictly available previous-week labels.

Uses the supplied hourly label files locally. Never upload them to GitHub.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)
FIRST, LAST = date(2025, 1, 1), date(2025, 10, 31)


def load(paths):
    count = (LAST - FIRST).days + 1
    values = np.zeros((count, len(ROUTES), 24), dtype=np.int64)
    occupied = set()
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source, delimiter=";"):
                day, route, hour = date.fromisoformat(row["date"]), int(row["route"]), int(row["hour"])
                if not FIRST <= day <= LAST or route not in ROUTES:
                    continue
                key = ((day - FIRST).days, ROUTES.index(route), hour)
                if key in occupied:
                    raise ValueError(f"duplicate label {key}")
                occupied.add(key)
                values[key] = int(row["boardings"])
    return values


def windows(kind, start, stop):
    begin, limit = (start - FIRST).days, (stop - FIRST).days
    if kind == "day":
        return [(day, 1) for day in range(begin, limit + 1)]
    # Monday cutoffs and seven-day, non-overlapping forecast intervals.
    return [(day, 7) for day in range(begin, limit - 5, 7)]


def measure(values, origins):
    weekdays = np.array([(FIRST + timedelta(days=i)).weekday() for i in range(len(values))])
    errors = {name: np.zeros(len(ROUTES), dtype=np.int64) for name in ("full_median", "previous_week")}
    actuals = np.zeros(len(ROUTES), dtype=np.int64)
    for cutoff, horizon in origins:
        if cutoff < 28 or cutoff + horizon > len(values):
            raise ValueError("invalid cutoff")
        history = values[:cutoff]
        by_hour = np.median(history, axis=0)
        by_weekday = {weekday: np.median(history[weekdays[:cutoff] == weekday], axis=0)
                      for weekday in range(7)}
        for lead in range(horizon):
            target = cutoff + lead
            lag = target - 7
            assert lag < cutoff, "future label used as predictor"
            truth = values[target]
            baseline = np.maximum(0, np.rint(.8 * by_weekday[weekdays[target]] + .2 * by_hour)).astype(np.int64)
            last_week = values[lag]
            for name, pred in (("full_median", baseline), ("previous_week", last_week)):
                errors[name] += np.abs(pred - truth).sum(axis=1)
            actuals += truth.sum(axis=1)
    total = int(actuals.sum())
    return {"origins": len(origins), "days": sum(h for _, h in origins), "actual_sum": total,
            "variants": {name: {"absolute_error_sum": int(err.sum()),
                                "wape_score": round(1 - int(err.sum()) / total, 6),
                                "error_by_route": {str(route): int(err[idx]) for idx, route in enumerate(ROUTES)}}
                         for name, err in errors.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forecast-cutoff", type=date.fromisoformat,
                        help="Optional last observed date, through 2025-10-31")
    parser.add_argument("--forecast-days", type=int, choices=(1, 7), default=7)
    parser.add_argument("--forecast-out", type=Path)
    args = parser.parse_args()
    values = load([args.train, args.test])
    intervals = {
        "day_jul_oct": ("day", date(2025, 7, 1), LAST),
        "day_sep_oct": ("day", date(2025, 9, 1), LAST),
        "week_jul_oct": ("week", date(2025, 7, 7), LAST),
        "week_sep_oct": ("week", date(2025, 9, 1), LAST),
    }
    result = {"method": "rolling as-of origins; full median versus last same weekday-hour; missing label = zero for competition WAPE only",
              "mode": "operational with daily/weekly observed label updates, not 61-day fixed competition submission",
              "routes": ROUTES,
              "folds": {key: measure(values, windows(*period)) for key, period in intervals.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, metrics in result["folds"].items():
        print(key, {k: v["wape_score"] for k, v in metrics["variants"].items()})
    if args.forecast_cutoff is not None:
        if args.forecast_out is None or not FIRST + timedelta(days=7) <= args.forecast_cutoff <= LAST:
            parser.error("forecast needs --forecast-out and an observed cutoff in Jan 8–Oct 31, 2025")
        begin = (args.forecast_cutoff - FIRST).days + 1
        args.forecast_out.parent.mkdir(parents=True, exist_ok=True)
        with args.forecast_out.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, delimiter=";")
            writer.writerow(("route", "date", "hour", "boardings"))
            for lead in range(args.forecast_days):
                date_string = (args.forecast_cutoff + timedelta(days=lead+1)).isoformat()
                for idx, route in enumerate(ROUTES):
                    for hour in range(24):
                        writer.writerow((route, date_string, hour, values[begin+lead-7, idx, hour]))
        print("forecast", args.forecast_out)


if __name__ == "__main__":
    main()
