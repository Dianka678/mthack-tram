#!/usr/bin/env python3
"""Аудит дневных сдвигов и пустых рабочих часов по готовым меткам.

Отчёт агрегирован, без идентификаторов поездок и пассажиров. Отсутствие строки
не доказывает нулевой спрос; предполагаемый интервал 06:00–22:00 служит только
для обнаружения аномалий и не заменяет историческое расписание.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    seen = set()
    daily = Counter()
    for source in (a.train, a.test):
        with open(source, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            if reader.fieldnames != ['route', 'date', 'hour', 'boardings']:
                raise ValueError(f'Unexpected schema: {source}: {reader.fieldnames}')
            for row in reader:
                route, day, hour = int(row['route']), date.fromisoformat(row['date']), int(row['hour'])
                key = (route, day, hour)
                if key in seen:
                    raise ValueError(f'Duplicate hour: {key}')
                seen.add(key)
                daily[route, day] += int(row['boardings'])
    routes = sorted({r for r, _, _ in seen})

    def days(start, end):
        d = date.fromisoformat(start)
        last = date.fromisoformat(end)
        while d <= last:
            yield d
            d += timedelta(days=1)

    def window(start, end):
        ds = list(days(start, end))
        seven = sum(daily[7, d] for d in ds)
        others = sum(daily[r, d] for r in routes if r != 7 for d in ds)
        return {'from': start, 'to': end, 'days': len(ds), 'route_7_total': seven,
                'other_routes_total': others, 'route_7_per_day': round(seven / len(ds), 1),
                'route_7_to_others': round(seven / others, 5)}

    missing_by_route_date = defaultdict(list)
    for d in days('2025-01-01', '2025-10-31'):
        for r in routes:
            for h in range(6, 23):
                if (r, d, h) not in seen:
                    missing_by_route_date[r, d.isoformat()].append(h)
    missing = [{'route': r, 'date': d, 'hours': hs}
               for (r, d), hs in sorted(missing_by_route_date.items())]
    report = {
        'method': 'Descriptive comparison only; no causal or out-of-sample effect estimated',
        'candidate_service_hours': '06:00–22:00; verify against historical schedules',
        'routes_in_labels': routes,
        'route_7_windows': {
            'before': window('2025-06-12', '2025-07-09'),
            'restriction': window('2025-07-10', '2025-08-10'),
            'after': window('2025-08-11', '2025-09-07'),
        },
        'missing_active_hours_count': sum(len(item['hours']) for item in missing),
        'missing_active_hours_by_route': dict(sorted(Counter(
            item['route'] for item in missing for _ in item['hours']).items())),
        'missing_active_hours_by_date': missing,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('route_7_windows',
                     'missing_active_hours_count', 'missing_active_hours_by_route')},
                     ensure_ascii=False, indent=2))
    print(f'Готово: {a.out}')


if __name__ == '__main__':
    main()
