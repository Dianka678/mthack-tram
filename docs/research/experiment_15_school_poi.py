"""Long-horizon school-term profile and as-of stop-link audit.

Run from anywhere:
python docs/research/experiment_15_school_poi.py --train labels_day_train.csv \
    --test labels_day_test.csv --stops route_reference.xlsx --out experiment_15.json

Dependency for the XLSX audit: openpyxl. Never publishes individual validations.
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

ROUTES = (1, 7, 11, 12, 17, 25, 26, 28, 50)
CUTOFF = date(2025, 8, 31)
VALID_START = date(2025, 9, 1)
VALID_END = date(2025, 10, 31)
AS_OF = ('2025-04-30', '2025-06-30', '2025-08-31', '2025-10-31')


def days(start, end):
    while start <= end:
        yield start
        start += timedelta(days=1)


def read_labels(paths):
    result = {}
    for path in paths:
        with open(path, encoding='utf-8-sig', newline='') as f:
            source = csv.DictReader(f, delimiter=';')
            if source.fieldnames != ['route', 'date', 'hour', 'boardings']:
                raise ValueError(f'Unexpected label fields: {path}')
            for row in source:
                key = (int(row['route']), date.fromisoformat(row['date']), int(row['hour']))
                if key in result or not (0 <= key[2] < 24):
                    raise ValueError(f'Duplicate/invalid label key: {key}')
                result[key] = int(row['boardings'])
    if {r for r, _, _ in result} != set(ROUTES) or max(d for _, d, _ in result) > VALID_END:
        raise ValueError('Unexpected routes or labels beyond validation period')
    return result


def profile_score(labels, exclude_summer_weekdays):
    local, hour = defaultdict(list), defaultdict(list)
    for d in days(date(2025, 1, 1), CUTOFF):
        if exclude_summer_weekdays and d.month in (7, 8) and d.weekday() < 5:
            continue
        for r in ROUTES:
            for h in range(24):
                value = labels.get((r, d, h), 0)
                local[r, d.weekday(), h].append(value)
                hour[r, h].append(value)
    local = {k: median(v) for k, v in local.items()}
    hour = {k: median(v) for k, v in hour.items()}
    error, total = 0, 0
    for d in days(VALID_START, VALID_END):
        for r in ROUTES:
            for h in range(24):
                actual = labels.get((r, d, h), 0)
                pred = round(.8 * local[r, d.weekday(), h] + .2 * hour[r, h])
                error += abs(actual - pred)
                total += actual
    return {'score': round(max(0, 1 - error / total), 6),
            'absolute_error': error, 'actual_sum': total}


def audited_stops(path):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book['Порядок_с_координатами']
    source = sheet.iter_rows(values_only=True)
    fields = next(source)
    names = {name: i for i, name in enumerate(fields)}
    required = ('route_short_name', 'stop_id', 'stop_name', 'start_date', 'actual_date')
    if any(x not in names for x in required):
        raise ValueError('Missing stop-reference fields')
    rows = []
    for values in source:
        try:
            route = int(values[names['route_short_name']])
        except (ValueError, TypeError):
            continue
        rows.append({'route': route, 'stop_id': values[names['stop_id']],
                     'name': str(values[names['stop_name']] or ''),
                     'start': str(values[names['start_date']] or '')[:10],
                     'actual': str(values[names['actual_date']] or '')[:10]})
    selected = [r for r in rows if r['route'] in ROUTES]
    book.close()
    return {
        'workbook_route_stop_rows': len(rows),
        'labeled_route_stop_rows': len(selected),
        'labeled_routes_with_future_reference': sorted({r['route'] for r in selected}),
        'valid_as_of': {cut: {'rows': len([r for r in selected if r['start'] and r['actual']
                                       and r['start'] <= cut and r['actual'] <= cut]),
                               'routes': sorted({r['route'] for r in selected if r['start'] and r['actual']
                                                 and r['start'] <= cut and r['actual'] <= cut})}
                        for cut in AS_OF},
        'stop_level_target_available': False,
        'poi_effect_estimable': False,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('train', 'test', 'stops', 'out'):
        cli.add_argument('--' + key, required=True, type=Path)
    a = cli.parse_args()
    labels = read_labels((a.train, a.test))
    base = profile_score(labels, False)
    school = profile_score(labels, True)
    out = {'cutoff': str(CUTOFF), 'holdout': [str(VALID_START), str(VALID_END)],
           'hour_grid': '9 routes × 61 dates × 24 hours; missing label=0 for metric only',
           'baseline': base, 'exclude_summer_weekdays': school,
           'delta_score': round(school['score'] - base['score'], 6),
           'selection_note': 'Exploratory: rule compared on this same holdout; no independent confirmation.',
           'stops': audited_stops(a.stops)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'baseline': base['score'], 'school': school['score'],
                      'valid_stop_rows_aug31': out['stops']['valid_as_of']['2025-08-31']['rows']}))


if __name__ == '__main__':
    main()
