#!/usr/bin/env python3
"""Парная абляция истории опубликованных апрельских ограничений маршрута 17."""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median

from backtest_baseline import dates, evaluate, load


RESTRICTED = {d for d in dates(date(2025, 4, 5), date(2025, 4, 27))
              if d.weekday() >= 5}
FOLDS = [('may_jun', date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
         ('jul_aug', date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31))]


def revised_route17_profile(data, cutoff, statistic):
    group, hour = defaultdict(list), defaultdict(list)
    for day in dates(date(2025, 1, 1), cutoff):
        if day in RESTRICTED:
            continue
        for h in range(24):
            y = data.get((17, day, h), 0)
            group[day.weekday(), h].append(y)
            hour[h].append(y)
    f = median if statistic == 'median' else lambda v: sum(v) / len(v)
    return ({key: f(v) for key, v in group.items()},
            {key: f(v) for key, v in hour.items()})


def metrics(predictions):
    buckets = defaultdict(lambda: [0, 0, 0])
    for r, _, _, actual, pred in predictions:
        for key in ('all', str(r)):
            buckets[key][0] += actual
            buckets[key][1] += abs(pred - actual)
            buckets[key][2] += pred - actual
    return {k: {'actual_sum': actual, 'absolute_error_sum': error,
                'pred_minus_actual': bias,
                'wape_score': round(max(0, 1 - error / actual), 6) if actual else None}
            for k, (actual, error, bias) in sorted(buckets.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    data = load(a.train) | load(a.test)
    routes = sorted({r for r, _, _ in data})
    report = {'sources': [
        {'published_at': '2025-04-04', 'url': 'https://transport.mos.ru/mostrans/all_news/123857'},
        {'published_at': '2025-04-22', 'url': 'https://transport.mos.ru/mostrans/all_news/124110'},
        {'published_at': '2025-04-30', 'url': 'https://www.mosmetro.ru/news/details/6793'}],
        'restricted_weekends_excluded_from_train': [d.isoformat() for d in sorted(RESTRICTED)],
        'effect_isolated_to_route': 17, 'folds': {}}
    for name, cutoff, first, last in FOLDS:
        history = {k: v for k, v in data.items() if k[1] <= cutoff}
        truth = {k: v for k, v in data.items() if first <= k[1] <= last}
        report['folds'][name] = {'cutoff': cutoff.isoformat(), 'validation':
                                  [first.isoformat(), last.isoformat()], 'variants': {}}
        for statistic in ('mean', 'median'):
            baseline, rows = evaluate(history, truth, routes, date(2025, 1, 1), cutoff,
                                      first, last, statistic)
            by_weekday_hour, by_hour = revised_route17_profile(history, cutoff, statistic)
            revised = [(r, day, h, y, max(0, round(.8 * by_weekday_hour[date.fromisoformat(day).weekday(), h]
                                             + .2 * by_hour[h]))) if r == 17
                       else (r, day, h, y, pred)
                       for r, day, h, y, pred in rows]
            report['folds'][name]['variants'][statistic] = {
                'baseline': metrics(rows), 'exclude_april_restrictions': metrics(revised),
                'untouched_other_routes': all(x == z for x, z in zip(rows, revised) if x[0] != 17)}
            print(name, statistic, 'all', baseline['all']['wape_score'],
                  report['folds'][name]['variants'][statistic]['exclude_april_restrictions']['all']['wape_score'],
                  '17', baseline['17']['wape_score'],
                  report['folds'][name]['variants'][statistic]['exclude_april_restrictions']['17']['wape_score'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
