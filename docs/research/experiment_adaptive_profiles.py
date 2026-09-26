#!/usr/bin/env python3
"""Парный опыт с давностью истории и официальными нерабочими днями.

Параметры выбираются только по May–Jun и Jul–Aug; Sep–Oct открывается
один раз для выбранного кандидата и двух зарегистрированных контролей.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from backtest_baseline import dates, fit, load
from experiment_calendar import NONWORK_WEEKDAYS


FOLDS = (
    ('may_jun', date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
    ('jul_aug', date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    ('sep_oct', date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
)
FIRST = date(2025, 1, 1)


def recent_means(history, routes, cutoff, half_life):
    local, hour, norm_local, norm_hour = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for r in routes:
        for day in dates(FIRST, cutoff):
            age = (cutoff - day).days
            weight = 2 ** (-age / half_life)
            for h in range(24):
                y = history.get((r, day, h), 0)
                a, b = (r, day.weekday(), h), (r, h)
                local[a] += weight * y
                hour[b] += weight * y
                norm_local[a] += weight
                norm_hour[b] += weight
    return ({k: v / norm_local[k] for k, v in local.items()},
            {k: v / norm_hour[k] for k, v in hour.items()})


def predict(models, route, day, hour, config):
    full, recent = models
    dow, by_hour, _ = full
    half_life, recent_weight, calendar = config

    def basic(day_of_week):
        full_local = dow[route, day_of_week, hour]
        full_base = .8 * full_local + .2 * by_hour[route, hour]
        if half_life is None:
            return full_base
        rec_dow, rec_hour = recent
        rec_base = .8 * rec_dow[route, day_of_week, hour] + .2 * rec_hour[route, hour]
        return (1 - recent_weight) * full_base + recent_weight * rec_base

    if calendar and day in NONWORK_WEEKDAYS:
        value = (basic(5) + basic(6)) / 2
    else:
        value = basic(day.weekday())
    return max(0, round(value))


def assess(data, routes, start, end, config, full, recent):
    totals = defaultdict(lambda: [0, 0])
    for route in routes:
        for day in dates(start, end):
            for h in range(24):
                truth = data.get((route, day, h), 0)
                value = predict((full, recent), route, day, h, config)
                for k in ('all', str(route)):
                    totals[k][0] += truth
                    totals[k][1] += abs(value - truth)
    return {k: {'actual_sum': v[0], 'absolute_error_sum': v[1],
                'wape_score': round(max(0, 1 - v[1] / v[0]), 6) if v[0] else None}
            for k, v in sorted(totals.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    data = load(args.train) | load(args.test)
    routes = sorted({r for r, _, _ in data})
    # Ограниченный список вариантов задан до вскрытия последнего среза.
    configs = {'mean_full': (None, 0, False),
               'median_full': (None, 0, False)}
    for hl in (28, 56, 112):
        for w in (.5, 1.0):
            configs[f'hl{hl}_w{w:g}'] = (hl, w, False)
            configs[f'hl{hl}_w{w:g}_calendar'] = (hl, w, True)
    report = {'hypothesis': 'recency weighted profile and prepublished holidays',
              'calendar_source': 'https://government.ru/docs/52895/',
              'candidate_grid': {k: {'half_life_days': v[0], 'recent_weight': v[1],
                                     'calendar': v[2]} for k, v in configs.items()},
              'missing_label_rule': 'zero for competition score, not a claim about actual demand',
              'folds': {}, 'selection': {}}
    for name, cutoff, start, end in FOLDS:
        is_gate = name == 'sep_oct'
        history = {k: v for k, v in data.items() if FIRST <= k[1] <= cutoff}
        if is_gate:
            errors = {config: sum(report['folds'][fold][config]['all']['absolute_error_sum']
                                  for fold in ('may_jun', 'jul_aug'))
                      for config in configs if config not in ('mean_full', 'median_full')}
            winner = min(errors, key=lambda k: (errors[k], k))
            report['selection'] = {'development_absolute_errors': errors,
                                   'candidate': winner, 'gate_not_untouched_final_test': True}
            evaluated = ('mean_full', 'median_full', winner)
        else:
            evaluated = tuple(configs)
        report['folds'][name] = {}
        required_hl = {configs[k][0] for k in evaluated if configs[k][0] is not None}
        caches = {hl: recent_means(history, routes, cutoff, hl) for hl in required_hl}
        for config in evaluated:
            statistic = 'median' if config == 'median_full' else 'mean'
            full = fit(history, routes, FIRST, cutoff, statistic)
            recent = caches.get(configs[config][0])
            result = assess(data, routes, start, end, configs[config], full, recent)
            report['folds'][name][config] = result
            print(name, config, result['all']['wape_score'], flush=True)
    base = min(('mean_full', 'median_full'),
               key=lambda k: report['folds']['sep_oct'][k]['all']['absolute_error_sum'])
    winner = report['selection']['candidate']
    report['selection']['gate_baseline'] = base
    report['selection']['gate_candidate_beats_baseline'] = (
        report['folds']['sep_oct'][winner]['all']['absolute_error_sum']
        < report['folds']['sep_oct'][base]['all']['absolute_error_sum'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('saved', args.out)


if __name__ == '__main__':
    main()
