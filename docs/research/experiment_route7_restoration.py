#!/usr/bin/env python3
"""Проверка известного к 13 августа изменения трассы маршрута 7.

Baseline обучается на всём 01.01–13.08. Candidate для маршрута 7 исключает
10.07–12.08, когда маршрут работал по укороченной трассе; для остальных
маршрутов и всех остальных дат обучение одинаковое. Прогноз: 14–31 августа.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median

from backtest_baseline import dates, fit, load


FIRST = date(2025, 1, 1)
CUTOFF = date(2025, 8, 13)
VALID_FROM = date(2025, 8, 14)
VALID_TO = date(2025, 8, 31)
SHORT_FROM = date(2025, 7, 10)
SHORT_TO = date(2025, 8, 12)


def fit_with_known_status(training, routes, statistic):
    by_dow_hour, by_hour, by_route = defaultdict(list), defaultdict(list), defaultdict(list)
    for route in routes:
        for day in dates(FIRST, CUTOFF):
            if route == 7 and SHORT_FROM <= day <= SHORT_TO:
                continue
            for hour in range(24):
                value = training.get((route, day, hour), 0)
                by_dow_hour[route, day.weekday(), hour].append(value)
                by_hour[route, hour].append(value)
                by_route[route].append(value)
    summarize = median if statistic == 'median' else lambda values: sum(values) / len(values)
    return ({k: summarize(v) for k, v in by_dow_hour.items()},
            {k: summarize(v) for k, v in by_hour.items()},
            {k: summarize(v) for k, v in by_route.items()})


def score(model, truth, routes):
    by_dow_hour, by_hour, by_route = model
    groups = defaultdict(lambda: {'actual_sum': 0, 'absolute_error_sum': 0,
                                  'pred_minus_actual': 0})
    for route in routes:
        for day in dates(VALID_FROM, VALID_TO):
            for hour in range(24):
                y = truth.get((route, day, hour), 0)
                local = by_dow_hour.get((route, day.weekday(), hour),
                                        by_hour.get((route, hour), by_route[route]))
                prediction = max(0, round(.8 * local + .2 * by_hour.get((route, hour), 0)))
                names = ['all', str(route)]
                if route == 7 and 6 <= hour <= 22:
                    names.append('route_7_daytime')
                for name in names:
                    m = groups[name]
                    m['actual_sum'] += y
                    m['absolute_error_sum'] += abs(y - prediction)
                    m['pred_minus_actual'] += prediction - y
    for m in groups.values():
        m['wape_score'] = round(max(0, 1 - m['absolute_error_sum'] / m['actual_sum']), 6) if m['actual_sum'] else None
    return dict(sorted(groups.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    data = load(a.train) | load(a.test)
    routes = sorted({r for r, _, _ in data})
    training = {k: v for k, v in data.items() if FIRST <= k[1] <= CUTOFF}
    truth = {k: v for k, v in data.items() if VALID_FROM <= k[1] <= VALID_TO}
    report = {
        'source_start': 'https://transport.mos.ru/mostrans/all_news/125272',
        'start_notice_published': '2025-07-09',
        'source_restored': 'https://transport.mos.ru/mostrans/all_news/125800',
        'restoration_notice_published': '2025-08-13',
        'cutoff': CUTOFF.isoformat(),
        'excluded_route_7_training_dates': [SHORT_FROM.isoformat(), SHORT_TO.isoformat()],
        'validation': [VALID_FROM.isoformat(), VALID_TO.isoformat()],
        'definition': 'Only route 7 closure-period training rows excluded; all other rows and models unchanged',
        'variants': {},
    }
    for statistic in ('mean', 'median'):
        baseline = fit(training, routes, FIRST, CUTOFF, statistic)
        adjusted = fit_with_known_status(training, routes, statistic)
        report['variants'][statistic] = {
            'base': score(baseline, truth, routes),
            'exclude_shortened_history': score(adjusted, truth, routes),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for method, variants in report['variants'].items():
        print(method, {k: {group: v[group]['wape_score']
                           for group in ('all', '7', 'route_7_daytime')}
                       for k, v in variants.items()})
    print('Готово:', a.out)


if __name__ == '__main__':
    main()
