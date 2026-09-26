#!/usr/bin/env python3
"""Парный опыт: официальные нерабочие будни получают профиль выходного.

Источник календаря: https://government.ru/docs/52895/ (04.10.2024).
Строит прогноз всего среза из данных ДО cutoff. Нерабочие субботы/воскресенья
сохраняют обычный профиль; только перенесённые/праздничные будни меняют его.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from backtest_baseline import dates, fit, load


NONWORK_WEEKDAYS = {
    # Нерабочие праздничные дни и перенесённые выходные, опубликованы в 2024.
    date.fromisoformat(s) for s in (
        '2025-01-01', '2025-01-02', '2025-01-03',
        '2025-01-06', '2025-01-07', '2025-01-08',
        '2025-05-01', '2025-05-02', '2025-05-08', '2025-05-09',
        '2025-06-12', '2025-06-13',
        '2025-11-03', '2025-11-04', '2025-12-31',
    )
}
FOLDS = {
    'may_jun': (date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
    'jul_aug': (date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    'sep_oct': (date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    values = load(args.train) | load(args.test)
    routes = sorted({r for r, _, _ in values})
    first = date(2025, 1, 1)
    report = {'source_url': 'https://government.ru/docs/52895/',
              'published_at': '2024-10-04',
              'hypothesis': 'Known non-working weekdays behave like weekends',
              'new_feature': 'official nonworking weekday; Saturday/Sunday average profile',
              'routes': routes, 'folds': {}}
    for fold, (cutoff, start, end) in FOLDS.items():
        history = {k: v for k, v in values.items() if first <= k[1] <= cutoff}
        eligible_days = [d for d in dates(start, end) if d in NONWORK_WEEKDAYS]
        variants = {}
        for statistic in ('mean', 'median'):
            by_dow_hour, by_hour, by_route = fit(history, routes, first, cutoff, statistic)
            accum = {key: {'actual_sum': 0, 'absolute_error_sum': 0,
                           'pred_minus_actual': 0, 'nonwork_absolute_error_sum': 0}
                     for key in ('base', 'calendar')}
            for r in routes:
                for d in dates(start, end):
                    for hour in range(24):
                        observed = values.get((r, d, hour), 0)
                        # Идентично функции evaluate в baseline.
                        local = by_dow_hour.get((r, d.weekday(), hour),
                                                 by_hour.get((r, hour), by_route[r]))
                        weekend = (by_dow_hour.get((r, 5, hour), by_hour[r, hour])
                                   + by_dow_hour.get((r, 6, hour), by_hour[r, hour])) / 2
                        for key, profile in (('base', local),
                                             ('calendar', weekend if d in NONWORK_WEEKDAYS else local)):
                            pred = max(0, round(.8 * profile + .2 * by_hour.get((r, hour), 0)))
                            m = accum[key]
                            m['actual_sum'] += observed
                            m['absolute_error_sum'] += abs(pred - observed)
                            m['pred_minus_actual'] += pred - observed
                            if d in NONWORK_WEEKDAYS:
                                m['nonwork_absolute_error_sum'] += abs(pred - observed)
            for v in accum.values():
                v['wape_score'] = round(max(0, 1 - v['absolute_error_sum'] / v['actual_sum']), 6)
            variants[statistic] = accum
        report['folds'][fold] = {
            'cutoff': cutoff.isoformat(), 'validation': [start.isoformat(), end.isoformat()],
            'eligible_nonworking_weekdays': [d.isoformat() for d in eligible_days],
            'n_changed_rows': len(eligible_days) * len(routes) * 24,
            'variants': variants,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    for fold, info in report['folds'].items():
        print(fold, 'changed', info['n_changed_rows'],
              {k: {mode: v['wape_score'] for mode, v in metrics.items()}
               for k, metrics in info['variants'].items()})
    print('Готово:', args.out)


if __name__ == '__main__':
    main()
