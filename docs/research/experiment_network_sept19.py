#!/usr/bin/env python3
"""Событийный backtest после публикаций 5 и 19 сентября 2025 года.

Калибровка маршрута 7 по июльскому перекрытию — только сценарий переноса
эффекта: сентябрьское перекрытие затрагивает другой участок трассы.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from backtest_baseline import dates, fit, load


CUTOFF = date(2025, 9, 19)
START, END = date(2025, 9, 20), date(2025, 9, 28)
EVENT_DATES = [d for d in dates(START, END) if d.weekday() >= 5]


def predict(profile, route, day, hour):
    by_weekday_hour, by_hour, _ = profile
    return max(0, round(.8 * by_weekday_hour[route, day.weekday(), hour]
                        + .2 * by_hour[route, hour]))


def route7_transfer_factor(labels):
    # Только субботы/воскресенья. Июльский период: опубликованное
    # укорочение 10.07–12.08; предшествующие 4 недели служат контролем.
    normal = [d for d in dates(date(2025, 6, 12), date(2025, 7, 9))
              if d.weekday() >= 5]
    shortened = [d for d in dates(date(2025, 7, 10), date(2025, 8, 10))
                 if d.weekday() >= 5]
    def daily_mean(days):
        return sum(labels.get((7, d, h), 0) for d in days for h in range(24)) / len(days)
    x, y = daily_mean(normal), daily_mean(shortened)
    return {'normal_weekend_mean': x, 'shortened_weekend_mean': y,
            'factor': y / x, 'normal_days': len(normal),
            'shortened_days': len(shortened)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    data = load(a.train) | load(a.test)
    routes = sorted({r for r, _, _ in data})
    history = {k: v for k, v in data.items() if k[1] <= CUTOFF}
    transfer = route7_transfer_factor(history)
    result = {'cutoff': CUTOFF.isoformat(), 'validation': [START.isoformat(), END.isoformat()],
              'event_dates': [d.isoformat() for d in EVENT_DATES],
              'news': {'route50_published': '2025-09-05',
                       'route50_url': 'https://mosmetro.ru/news/details/7732',
                       'route7_published': '2025-09-19',
                       'route7_url': 'https://transport.mos.ru/mostrans/all_news/126405'},
              'route7_scenario_transfer': transfer,
              'training_rule': 'missing labels filled zero; full history 2025-01-01 to cutoff',
              'predictions': {}, 'metrics': {}}
    for kind in ('mean', 'median'):
        profile = fit(history, routes, date(2025, 1, 1), CUTOFF, kind)
        sums = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for route in routes:
            for day in dates(START, END):
                for hour in range(24):
                    baseline = predict(profile, route, day, hour)
                    truth = data.get((route, day, hour), 0)
                    candidate = baseline
                    if route == 50 and day in EVENT_DATES:
                        candidate = 0
                    transferred = candidate
                    if route == 7 and day in EVENT_DATES:
                        transferred = round(candidate * transfer['factor'])
                    for name, pred in [('baseline', baseline), ('shutdown_50', candidate),
                                       ('shutdown_50_and_7_transfer_scenario', transferred)]:
                        for key in ('all', str(route)):
                            sums[name][key][0] += truth
                            sums[name][key][1] += abs(pred - truth)
                    if route in (7, 50) and day in EVENT_DATES:
                        row = result['predictions'].setdefault(kind, {}).setdefault(str(route), {})
                        bucket = row.setdefault(day.isoformat(), {'actual': 0, 'baseline': 0,
                            'shutdown_50': 0, 'shutdown_50_and_7_transfer_scenario': 0})
                        bucket['actual'] += truth
                        bucket['baseline'] += baseline
                        bucket['shutdown_50'] += candidate
                        bucket['shutdown_50_and_7_transfer_scenario'] += transferred
        result['metrics'][kind] = {name: {route: {
            'actual_sum': v[0], 'absolute_error_sum': v[1],
            'wape_score': round(max(0, 1 - v[1] / v[0]), 6) if v[0] else None}
            for route, v in totals.items()} for name, totals in sums.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for kind, variants in result['metrics'].items():
        print(kind, {k: (v['all']['wape_score'], v['7']['wape_score'],
                         v['50']['wape_score']) for k, v in variants.items()})


if __name__ == '__main__':
    main()
