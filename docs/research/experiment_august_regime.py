#!/usr/bin/env python3
"""As-of Aug-31 transport notice ablation on a fixed Sep–Oct hourly grid.

Exploratory: the validation interval has already been used in prior research.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import median

from backtest_baseline import dates, load


def run(train, truth):
    routes = sorted({route for route, _, _ in train})
    ordinary = defaultdict(list)
    hourly = defaultdict(list)
    recent = defaultdict(list)
    for route in routes:
        for day in dates(date(2025, 1, 1), date(2025, 8, 31)):
            for hour in range(24):
                value = train.get((route, day, hour), 0)
                if day.month not in (7, 8) or day.weekday() >= 5:
                    ordinary[route, day.weekday(), hour].append(value)
                    hourly[route, hour].append(value)
                # Official Aug 13 restoration of regular service on routes 7, 50.
                # The route 7 weekend detour was published Aug 15 and began Aug 16.
                start = date(2025, 8, 16) if route == 7 and day.weekday() >= 5 else date(2025, 8, 13)
                if route in (7, 50) and day >= start:
                    recent[route, day.weekday(), hour].append(value)
    metrics = {variant: defaultdict(int) for variant in
               ('school_profile', 'aug7_weekends', 'aug7_all', 'aug7_50_all')}
    impacted = {variant: defaultdict(int) for variant in metrics}
    for route in routes:
        for day in dates(date(2025, 9, 1), date(2025, 10, 31)):
            for hour in range(24):
                key = (route, day, hour)
                truth_value = truth.get(key, 0)
                base = 0.8 * median(ordinary[route, day.weekday(), hour]) + 0.2 * median(hourly[route, hour])
                # Predeclared 25% shrink of sparse post-restoration route-hour
                # profile. We do not learn this weight from validation labels.
                new = median(recent[route, day.weekday(), hour]) if recent[route, day.weekday(), hour] else base
                for variant in metrics:
                    apply = ((variant == 'aug7_weekends' and route == 7 and day.weekday() >= 5)
                             or (variant == 'aug7_all' and route == 7)
                             or (variant == 'aug7_50_all' and route in (7, 50)))
                    prediction = max(0, round(.75 * base + .25 * new if apply else base))
                    for group in ('all', str(route)):
                        metrics[variant][group] += abs(prediction - truth_value)
                        impacted[variant][group] += truth_value
    return {variant: {key: {'absolute_error_sum': error,
                            'actual_sum': impacted[variant][key],
                            'wape_score': round(1-error / impacted[variant][key], 6)}
                      for key, error in groups.items()} for variant, groups in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    train, truth = load(args.train), load(args.test)
    assert all(day <= date(2025, 8, 31) for _, day, _ in train)
    result = {'cutoff': '2025-08-31', 'validation': '2025-09-01/2025-10-31',
              'status': 'exploratory; validation previously viewed',
              'source': {'restored': 'https://transport.mos.ru/mostrans/all_news/125800',
                         'weekend_detour': 'https://transport.mos.ru/mostrans/all_news/125846'},
              'metrics': run(train, truth)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, groups in result['metrics'].items():
        print(name, groups['all']['wape_score'], groups['7']['absolute_error_sum'], groups['50']['absolute_error_sum'])


if __name__ == '__main__':
    main()
