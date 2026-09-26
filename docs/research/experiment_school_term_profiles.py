#!/usr/bin/env python3
"""Exploratory as-of-cutoff calendar regimes for long hourly tram forecasts.

The four filters are compared on the same Sep–Oct holdout. Choosing the best
filter with this holdout makes its score exploratory, not a final test score.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median

from backtest_baseline import dates, evaluate, load


FILTERS = {
    "exclude_jul_aug_all": lambda d: d.month not in (7, 8),
    "exclude_jul_aug_weekdays": lambda d: d.month not in (7, 8) or d.weekday() >= 5,
    "only_feb_apr": lambda d: d.month in (2, 3, 4),
    "only_feb_jun": lambda d: d.month in (2, 3, 4, 5, 6),
}


def score(train, truth, routes, keep):
    by_dow_hour, by_hour = defaultdict(list), defaultdict(list)
    for route in routes:
        for day in dates(date(2025, 1, 1), date(2025, 8, 31)):
            if not keep(day):
                continue
            for hour in range(24):
                value = train.get((route, day, hour), 0)
                by_dow_hour[route, day.weekday(), hour].append(value)
                by_hour[route, hour].append(value)
    error, actual, bias = defaultdict(int), defaultdict(int), defaultdict(int)
    for route in routes:
        for day in dates(date(2025, 9, 1), date(2025, 10, 31)):
            for hour in range(24):
                value = truth.get((route, day, hour), 0)
                pred = max(0, round(0.8 * median(by_dow_hour[route, day.weekday(), hour])
                                    + 0.2 * median(by_hour[route, hour])))
                error[route] += abs(pred - value)
                actual[route] += value
                bias[route] += pred - value
    result = {str(route): {"absolute_error_sum": error[route],
                           "actual_sum": actual[route],
                           "pred_minus_actual": bias[route],
                           "wape_score": round(1 - error[route] / actual[route], 6)}
              for route in routes}
    result["all"] = {"absolute_error_sum": sum(error.values()),
                     "actual_sum": sum(actual.values()),
                     "pred_minus_actual": sum(bias.values()),
                     "wape_score": round(1 - sum(error.values()) / sum(actual.values()), 6)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    train, truth = load(args.train), load(args.test)
    assert all(day <= date(2025, 8, 31) for _, day, _ in train)
    routes = sorted({route for route, _, _ in train})
    base, _ = evaluate(train, truth, routes, date(2025, 1, 1),
                       date(2025, 8, 31), date(2025, 9, 1),
                       date(2025, 10, 31), "median")
    results = {name: score(train, truth, routes, keep)
               for name, keep in FILTERS.items()}
    payload = {"train_cutoff": "2025-08-31", "validation": "2025-09-01/2025-10-31",
               "rule": "Missing hourly label means zero only for scoring, as in baseline; same 9 routes and all 24 hours.",
               "selection_warning": "Four alternatives compared on validation; winner is exploratory and cannot be claimed as independently validated.",
               "baseline_median": base, "variants": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("baseline", base["all"])
    for name, result in results.items():
        print(name, result["all"])


if __name__ == "__main__":
    main()
