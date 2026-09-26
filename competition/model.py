"""Standalone hourly tram forecast, 2025 Nov–Dec competition submission.

python tram_ds2_school_regime_20260926.py --train labels_day_train.csv \
  --test labels_day_test.csv --sample test_submission.csv --out submission.csv

The 0.886109 score is exploratory Sep–Oct 2025 (route 7 decay tuned on it).
No Nov–Dec target is used. Route 5 is cold-start zero due to missing history.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)
TARGET_ROUTES = tuple(sorted((*ROUTES, 5)))
FIRST = date(2025, 1, 1)
REGIME_START = date(2025, 8, 16)
REGIME_ANCHOR = date(2025, 8, 31)
DECAY_DAYS = 120
WORKING = {date(2025, 11, 1)}
NONWORKING = {date(2025, 11, 3), date(2025, 11, 4), date(2025, 12, 31)}


def read_labels(paths):
    result = {}
    for path in paths:
        with open(path, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            if reader.fieldnames != ['route', 'date', 'hour', 'boardings']:
                raise ValueError(f'Unexpected label fields: {path}: {reader.fieldnames}')
            for row in reader:
                r, d, h = int(row['route']), date.fromisoformat(row['date']), int(row['hour'])
                key = (r, d, h)
                if key in result or not (0 <= h <= 23):
                    raise ValueError(f'Duplicate or invalid key: {key}')
                result[key] = int(row['boardings'])
    return result


def days(first, last):
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


def fit(history, cutoff):
    if any(d > cutoff for _, d, _ in history):
        history = {k: v for k, v in history.items() if k[1] <= cutoff}
    local, by_hour, regime = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in days(FIRST, cutoff):
        for r in ROUTES:
            for h in range(24):
                value = history.get((r, d, h), 0)
                if d.month not in (7, 8) or d.weekday() >= 5:
                    local[r, d.weekday(), h].append(value)
                    by_hour[r, h].append(value)
                if r == 7 and d >= REGIME_START and d.weekday() >= 5:
                    regime[d.weekday(), h].append(value)
    return tuple({k: median(values) for k, values in group.items()}
                 for group in (local, by_hour, regime))


def forecast(key, model, calendar=True):
    r, d, h = key
    if r == 5:
        return 0
    local, by_hour, regime = model
    day_type = (0 if d in WORKING else 6 if d in NONWORKING else d.weekday()) if calendar else d.weekday()
    parent = by_hour.get((r, h), 0)
    base = .8 * local.get((r, day_type, h), parent) + .2 * parent
    if r == 7 and d.weekday() >= 5 and d >= REGIME_ANCHOR:
        weight = max(0, 1 - (d - REGIME_ANCHOR).days / DECAY_DAYS)
        base = (1 - weight) * base + weight * regime.get((d.weekday(), h), base)
    return max(0, round(base))


def grid(routes, first, last):
    return {(r, d, h) for r in routes for d in days(first, last) for h in range(24)}


def score(history):
    cutoff, first, last = date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)
    model = fit(history, cutoff)
    keys = grid(ROUTES, first, last)
    total = sum(history.get(k, 0) for k in keys)
    error = sum(abs(history.get(k, 0) - forecast(k, model, calendar=False)) for k in keys)
    return round(1 - error / total, 6)


def save_model(path, model, inputs):
    local, by_hour, regime = model
    payload = {
        'name': 'ds2_school_regime_20260926', 'as_of': '2025-10-31',
        'routes': list(ROUTES), 'cold_start_route': 5,
        'regime_start': str(REGIME_START), 'regime_anchor': str(REGIME_ANCHOR),
        'decay_days': DECAY_DAYS, 'local_weight': 0.8,
        'source_sha256': {name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                          for name, source in inputs.items()},
        'profiles': {
            'local': [[*key, value] for key, value in sorted(local.items())],
            'hour': [[*key, value] for key, value in sorted(by_hour.items())],
            'regime': [[*key, value] for key, value in sorted(regime.items())],
        },
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n',
                          encoding='utf-8')


def load_model(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if (payload.get('name') != 'ds2_school_regime_20260926'
            or payload.get('as_of') != '2025-10-31'
            or payload.get('routes') != list(ROUTES)
            or payload.get('regime_anchor') != str(REGIME_ANCHOR)
            or payload.get('decay_days') != DECAY_DAYS):
        raise ValueError('Incompatible model artifact')
    return tuple({tuple(row[:-1]): row[-1] for row in payload['profiles'][name]}
                 for name in ('local', 'hour', 'regime'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'test', 'sample', 'out', 'model-in', 'model-out'):
        p.add_argument('--' + key, type=Path, required=key in ('sample', 'out'))
    a = p.parse_args()
    if a.model_in:
        model = load_model(a.model_in)
        history = None
    else:
        if not a.train or not a.test:
            p.error('Either --model-in or both --train and --test are required')
        history = read_labels((a.train, a.test))
        if {r for r, _, _ in history} != set(ROUTES) or any(d > date(2025, 10, 31) for _, d, _ in history):
            raise ValueError('Unexpected routes or labels beyond 2025-10-31')
        model = fit(history, date(2025, 10, 31))
    if not a.sample or not a.out:
        p.error('--sample and --out are required')
    with open(a.sample, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if reader.fieldnames != ['route', 'date', 'hour', 'prediction']:
            raise ValueError(f'Unexpected sample fields: {reader.fieldnames}')
        template = list(reader)
    keys = [(int(x['route']), date.fromisoformat(x['date']), int(x['hour'])) for x in template]
    expected = grid(TARGET_ROUTES, date(2025, 11, 1), date(2025, 12, 31))
    if len(keys) != 14640 or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Incomplete template grid')
    if a.model_out:
        if history is None:
            raise ValueError('--model-out requires --train and --test')
        save_model(a.model_out, model, {'train': a.train, 'test': a.test})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['route', 'date', 'hour', 'prediction'],
                                delimiter=';', lineterminator='\n')
        writer.writeheader()
        for source, key in zip(template, keys):
            writer.writerow({**source, 'prediction': forecast(key, model)})
    print(json.dumps({'local_sep_oct_score_exploratory': score(history) if history else None,
                      'rows': len(keys), 'output': str(a.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
