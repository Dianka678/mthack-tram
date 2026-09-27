#!/usr/bin/env python3
"""Calendar-only annual harmonic on top of the median hourly baseline.

Exploratory screen: the same three ridge values are shown for all folds.
The target fold is never used to fit coefficients, although reporting all
candidate scores means it should not be used for later model selection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median

import numpy as np

from backtest_baseline import dates, fit, load


FOLDS = [("may_jun", date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
         ("jul_aug", date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
         ("sep_oct", date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31))]


def vector(day):
    t = 2 * np.pi * day.timetuple().tm_yday / 365
    return np.array([1.0, np.sin(t), np.cos(t)])


def run_fold(all_data, routes, core, cutoff, begin, end):
    train = {key: value for key, value in all_data.items() if key[1] <= cutoff}
    dow_hour, hour_profile, _ = fit(train, routes, date(2025, 1, 1), cutoff, "median")
    days = list(dates(date(2025, 1, 1), cutoff))
    network = {day: sum(train.get((route, day, hour), 0)
                        for route in core for hour in range(24)) for day in days}
    grouped = defaultdict(list)
    for day in days:
        grouped[day.weekday()].append(network[day])
    weekday_median = {weekday: median(values) for weekday, values in grouped.items()}
    x = np.array([vector(day) for day in days])
    y = np.array([network[day] / weekday_median[day.weekday()] for day in days])
    results = {}
    for penalty in (0, 10, 50):
        beta = np.linalg.solve(x.T @ x + np.diag([0, penalty, penalty]), x.T @ y)
        error = baseline_error = actual_sum = 0
        for route in routes:
            for day in dates(begin, end):
                factor = float(np.clip(vector(day) @ beta, 0.7, 1.3))
                for hour in range(24):
                    true = all_data.get((route, day, hour), 0)
                    profile = max(0, round(0.8 * dow_hour[route, day.weekday(), hour]
                                            + 0.2 * hour_profile[route, hour]))
                    error += abs(true - round(profile * factor))
                    baseline_error += abs(true - profile)
                    actual_sum += true
        results[str(penalty)] = {
            "coefficients_intercept_sin_cos": [round(float(z), 6) for z in beta],
            "baseline_score": round(1 - baseline_error / actual_sum, 6),
            "candidate_score": round(1 - error / actual_sum, 6),
            "baseline_absolute_error": baseline_error,
            "candidate_absolute_error": error,
            "actual_sum": actual_sum,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    all_data = load(args.train) | load(args.test)
    routes = sorted({route for route, _, _ in all_data})
    core = [route for route in routes if route not in (7, 50)]
    payload = {"calendar_features": ["sin(2*pi*day_of_year/365)",
                                     "cos(2*pi*day_of_year/365)"],
               "network_scale_routes": core,
               "excluded_from_scale": [7, 50],
               "rules": "Regress aggregate pre-cutoff daily count / pre-cutoff weekday median; multiply median hourly baseline by clipped factor [0.7,1.3]. No future labels in regression.",
               "folds": {name: run_fold(all_data, routes, core, cutoff, begin, end)
                         for name, cutoff, begin, end in FOLDS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, variants in payload["folds"].items():
        print(name, [(k, v["baseline_score"], v["candidate_score"])
                     for k, v in variants.items()])


if __name__ == "__main__":
    main()
