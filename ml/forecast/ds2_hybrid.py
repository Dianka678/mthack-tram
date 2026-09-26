"""DS-2 informed hourly tram forecast, using only labels available by cutoff.

Usage: python ds2_informed_model.py --train TRAIN --test TEST --sample SAMPLE --out CSV
No actual November/December validations enter fit or model selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

HOLIDAYS_2025 = {
    date.fromisoformat(s) for s in (
        '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04',
        '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08',
        '2025-02-23', '2025-03-08', '2025-05-01', '2025-05-02',
        '2025-05-08', '2025-05-09', '2025-06-12', '2025-06-13',
        '2025-11-03', '2025-11-04', '2025-12-31',
    )
}
WORKING_SATURDAYS_2025 = {date(2025, 11, 1)}
FIRST = date(2025, 1, 1)


def load_labels(paths):
    data = {}
    for path in paths:
        with open(path, encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter=';')
            if reader.fieldnames != ['route', 'date', 'hour', 'boardings']:
                raise ValueError(f'Unexpected labels schema: {path}')
            for row in reader:
                key = (row['route'], date.fromisoformat(row['date']), int(row['hour']))
                if key in data:
                    raise ValueError(f'Duplicate label {key}')
                data[key] = int(row['boardings'])
    return data


def all_keys(routes, start, end):
    day = start
    while day <= end:
        for route in routes:
            for hour in range(24):
                yield route, day, hour
        day += timedelta(days=1)


def day_type(day, calendar=True):
    if not calendar:
        return day.weekday()
    if day in WORKING_SATURDAYS_2025:
        return 0  # proxy working Monday; known Saturday workday
    if day in HOLIDAYS_2025:
        return 6  # Sunday proxy for nonworking public holiday
    return day.weekday()


def fit(history, cutoff, calendar=True, route50_regime=False):
    routes = sorted({r for r, d, h in history if d <= cutoff}, key=int)
    local, hour = defaultdict(list), defaultdict(list)
    recent50 = defaultdict(list)
    for r, d, h in all_keys(routes, FIRST, cutoff):
        value = history.get((r, d, h), 0)  # same missing-label policy as DS-1
        local[r, day_type(d, calendar), h].append(value)
        hour[r, h].append(value)
        if route50_regime and r == '50' and d >= date(2025, 9, 6) and d.weekday() >= 5:
            recent50[d.weekday(), h].append(value)
    return ({k: median(v) for k, v in local.items()},
            {k: median(v) for k, v in hour.items()},
            {k: median(v) for k, v in recent50.items()})


def forecast(keys, model, calendar=True, route50_regime=False):
    local, hour, recent50 = model
    result = {}
    for r, d, h in keys:
        parent = hour.get((r, h), 0)
        baseline = .8 * local.get((r, day_type(d, calendar), h), parent) + .2 * parent
        if route50_regime and r == '50' and d.weekday() >= 5 and d >= date(2025, 9, 6):
            # Empirical post-change weekend profile. No hardcoded zero: some
            # September/October weekends still have valid boarding records.
            baseline = recent50.get((d.weekday(), h), baseline)
        result[(r, d, h)] = max(0, round(baseline))
    return result


def score(truth, pred, keys):
    keys = list(keys)
    actual = sum(truth.get(key, 0) for key in keys)
    error = sum(abs(truth.get(key, 0) - pred[key]) for key in keys)
    return {'rows': len(keys), 'actual_sum': actual, 'absolute_error': error,
            'score': max(0, 1 - error / actual) if actual else None}


def main():
    p = argparse.ArgumentParser()
    for arg in ('train', 'test', 'sample', 'out'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--experimental-route50', action='store_true',
                   help='Extrapolate the September–October weekend regime to Nov–Dec; requires schedule confirmation')
    args = p.parse_args()
    labels = load_labels([args.train, args.test])
    routes = sorted({r for r, _, _ in labels}, key=int)
    audit = {}
    # DS-2's calendar signal on the only earlier fold containing public holidays.
    for calendar in (False, True):
        model = fit(labels, date(2025, 4, 30), calendar=calendar)
        keys = list(all_keys(routes, date(2025, 5, 1), date(2025, 6, 30)))
        audit['may_jun_calendar_' + str(calendar)] = score(labels, forecast(keys, model, calendar), keys)
    # Post-change route 50 weekend profile is learned from September and tested on October.
    for regime in (False, True):
        model = fit(labels, date(2025, 9, 30), route50_regime=regime)
        keys = list(all_keys(routes, date(2025, 10, 1), date(2025, 10, 31)))
        pred = forecast(keys, model, route50_regime=regime)
        audit['october_regime_' + str(regime)] = score(labels, pred, keys)
        audit['october_route50_regime_' + str(regime)] = score(labels, pred, (k for k in keys if k[0] == '50'))
    # DS-2 has not verified the route 50 weekend schedule for all of Nov–Dec.
    # The default competition model keeps calendar with the full-history profile.
    model = fit(labels, date(2025, 10, 31), route50_regime=args.experimental_route50)
    with args.sample.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        if reader.fieldnames != ['route', 'date', 'hour', 'prediction']:
            raise ValueError('Unexpected sample schema')
        rows = list(reader)
    keys = [(row['route'], date.fromisoformat(row['date']), int(row['hour'])) for row in rows]
    expected = set(all_keys(sorted({r for r, _, _ in keys}, key=int), date(2025, 11, 1), date(2025, 12, 31)))
    if len(keys) != 14640 or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Incomplete or duplicated sample grid')
    pred = forecast(keys, model, route50_regime=args.experimental_route50)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, delimiter=';', lineterminator='\n')
        writer.writeheader()
        for row, key in zip(rows, keys):
            row['prediction'] = pred[key]
            writer.writerow(row)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
