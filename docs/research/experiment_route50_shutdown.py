#!/usr/bin/env python3
"""Проверка известной к 5 сентября отмены маршрута 50 по выходным."""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from backtest_baseline import dates, fit, load


START = date(2025, 1, 1)
CUTOFF = date(2025, 9, 5)
BEGIN = date(2025, 9, 6)
END = date(2025, 9, 28)


def compare(model, data, routes):
    by_dow_hour, by_hour, by_route = model
    groups = {mode: defaultdict(lambda: {'actual_sum': 0, 'absolute_error_sum': 0,
                                         'pred_minus_actual': 0})
              for mode in ('base', 'shutdown')}
    changed_days = defaultdict(lambda: {'actual_sum': 0, 'base_prediction_sum': 0,
                                         'positive_hours': 0})
    for route in routes:
        for day in dates(BEGIN, END):
            for hour in range(24):
                actual = data.get((route, day, hour), 0)
                local = by_dow_hour.get((route, day.weekday(), hour),
                                        by_hour.get((route, hour), by_route[route]))
                baseline = max(0, round(.8 * local + .2 * by_hour.get((route, hour), 0)))
                announced = route == 50 and day.weekday() >= 5
                if announced:
                    change = changed_days[day.isoformat()]
                    change['actual_sum'] += actual
                    change['base_prediction_sum'] += baseline
                    change['positive_hours'] += actual > 0
                for mode, prediction in (('base', baseline),
                                         ('shutdown', 0 if announced else baseline)):
                    names = ['all', str(route)]
                    if announced:
                        names.extend(('route_50_weekends', 'route_50_weekends_daytime')
                                     if 6 <= hour <= 22 else ('route_50_weekends',))
                    for name in names:
                        row = groups[mode][name]
                        row['actual_sum'] += actual
                        row['absolute_error_sum'] += abs(actual - prediction)
                        row['pred_minus_actual'] += prediction - actual
    for mode in groups:
        for row in groups[mode].values():
            row['wape_score'] = round(max(0, 1 - row['absolute_error_sum'] / row['actual_sum']), 6) if row['actual_sum'] else None
        groups[mode] = dict(sorted(groups[mode].items()))
    return groups, dict(sorted(changed_days.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    data = load(args.train) | load(args.test)
    routes = sorted({route for route, _, _ in data})
    training = {key: value for key, value in data.items() if START <= key[1] <= CUTOFF}
    report = {'source': 'https://mosmetro.ru/news/details/7732',
              'published_at': CUTOFF.isoformat(), 'cutoff': CUTOFF.isoformat(),
              'validation': [BEGIN.isoformat(), END.isoformat()],
              'missing_label_rule': 'zero only for competition WAPE; positive labels retained',
              'changed_rows_per_statistic': sum(day.weekday() >= 5 for day in dates(BEGIN, END)) * 24,
              'variants': {}}
    for statistic in ('mean', 'median'):
        groups, days = compare(fit(training, routes, START, CUTOFF, statistic), data, routes)
        report['variants'][statistic] = groups
        report['affected_day_facts'] = days
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for statistic, variants in report['variants'].items():
        print(statistic, {mode: {group: result[group]['wape_score']
                                 for group in ('all', '50', 'route_50_weekends')}
                          for mode, result in variants.items()})
    print('Готово:', args.out)


if __name__ == '__main__':
    main()
