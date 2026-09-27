#!/usr/bin/env python3
"""Воспроизводимый календарный baseline на агрегированных метках.

Оценивает весь маршрут × дата × час, отсутствующие в labels часы считает нулями
ТОЛЬКО для конкурсной метрики. Отсутствие строки само по себе не доказывает
нулевой спрос: диагностика сбоев делается по сырым данным отдельно.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median


def load(path: str):
    data = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f, delimiter=";")
        if set(rows.fieldnames or []) != {"route", "date", "hour", "boardings"}:
            raise ValueError(f"Неверная схема {path}: {rows.fieldnames}")
        for row in rows:
            key = (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
            if key in data:
                raise ValueError(f"Повтор ключа: {key}")
            data[key] = int(row["boardings"])
    return data


def dates(a: date, b: date):
    while a <= b:
        yield a
        a += timedelta(days=1)


def fit(train, routes, start, end, statistic):
    rdh, rh, r = defaultdict(list), defaultdict(list), defaultdict(list)
    for route in routes:
        for day in dates(start, end):
            for hour in range(24):
                value = train.get((route, day, hour), 0)
                rdh[route, day.weekday(), hour].append(value)
                rh[route, hour].append(value)
                r[route].append(value)
    f = median if statistic == "median" else lambda a: sum(a) / len(a)
    return ({k: f(v) for k, v in rdh.items()},
            {k: f(v) for k, v in rh.items()},
            {k: f(v) for k, v in r.items()})


def evaluate(train, truth, routes, train_start, train_end, valid_start, valid_end, statistic):
    by_dow_hour, by_hour, by_route = fit(train, routes, train_start, train_end, statistic)
    mae = defaultdict(float)
    totals = defaultdict(int)
    bias = defaultdict(float)
    predictions = []
    for route in routes:
        for day in dates(valid_start, valid_end):
            for hour in range(24):
                key = (route, day, hour)
                actual = truth.get(key, 0)
                # Сглаживание предотвращает шум редкой группы, сохраняет нули ночью.
                local = by_dow_hour.get((route, day.weekday(), hour),
                                        by_hour.get((route, hour), by_route[route]))
                pred = max(0, round(0.8 * local + 0.2 * by_hour.get((route, hour), 0)))
                mae["all"] += abs(actual - pred)
                mae[str(route)] += abs(actual - pred)
                totals["all"] += actual
                totals[str(route)] += actual
                bias["all"] += pred - actual
                bias[str(route)] += pred - actual
                predictions.append((route, day.isoformat(), hour, actual, pred))
    metrics = {k: {"wape_score": round(max(0, 1 - mae[k] / totals[k]), 6)
                    if totals[k] else None,
                    "actual_sum": totals[k], "absolute_error_sum": round(mae[k]),
                    "pred_minus_actual": round(bias[k])}
               for k in sorted(totals)}
    return metrics, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_data = load(a.train) | load(a.test)
    routes = sorted({r for r, _, _ in all_data})
    folds = [
        ("jul_aug", date(2025, 1, 1), date(2025, 6, 30),
         date(2025, 7, 1), date(2025, 8, 31)),
        ("sep_oct", date(2025, 1, 1), date(2025, 8, 31),
         date(2025, 9, 1), date(2025, 10, 31)),
    ]
    report = {"method": "route × weekday × hour, 20% shrink toward route × hour",
              "missing_label_rule": "filled zero for score; investigate outages separately",
              "routes": routes, "folds": {}}
    for name, start, cutoff, begin, end in folds:
        report["folds"][name] = {}
        for statistic in ("mean", "median"):
            train = {k: v for k, v in all_data.items() if start <= k[1] <= cutoff}
            truth = {k: v for k, v in all_data.items() if begin <= k[1] <= end}
            metrics, _ = evaluate(train, truth, routes, start, cutoff, begin, end, statistic)
            report["folds"][name][statistic] = metrics
    (a.out / "backtest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    for fold, variants in report["folds"].items():
        print(fold, *(f"{name}={value['all']['wape_score']:.4f}"
                      for name, value in variants.items()))
    print(f"Готово: {a.out / 'backtest.json'}")


if __name__ == "__main__":
    main()
