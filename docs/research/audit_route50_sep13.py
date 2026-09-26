#!/usr/bin/env python3
"""Потоковая сверка аномальных валидаций маршрута 50 за 13 сентября.

Читает test.zip::test.csv, выводит только агрегаты без идентификаторов карт
и устройств. Время берётся из tran_date_time.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from profile_data import open_text, parse_route, parse_timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raw = Counter()
    types = Counter()
    day_status = Counter()
    invalid_time = 0
    read_rows = 0
    cache = {}
    with open_text(args.input) as handle:
        reader = csv.DictReader(handle, delimiter=';')
        required = {'tran_date_time', 'validation_result', 'ngpt_route', 'tran_type_id'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing input columns: {required - set(reader.fieldnames or [])}')
        for row in reader:
            read_rows += 1
            if parse_route(row['ngpt_route']) != 50:
                continue
            when = parse_timestamp(row['tran_date_time'], cache)
            if when is None:
                invalid_time += 1
                continue
            day, hour = when
            if not '2025-09-06' <= day <= '2025-09-28':
                continue
            status = row['validation_result']
            if day == '2025-09-13':
                day_status[status] += 1
                types[(status, row['tran_type_id'])] += 1
            if status == '1':
                raw[(day, hour)] += 1

    labels = Counter()
    with open(args.labels, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle, delimiter=';'):
            if row['route'] == '50' and '2025-09-06' <= row['date'] <= '2025-09-28':
                labels[(row['date'], int(row['hour']))] += int(row['boardings'])
    sep13 = []
    for hour in range(24):
        key = ('2025-09-13', hour)
        sep13.append({'hour': hour, 'raw_success': raw[key], 'label': labels[key],
                      'difference': raw[key] - labels[key]})
    days = sorted({day for day, _ in raw} | {day for day, _ in labels})
    report = {'source': args.input, 'rows_scanned': read_rows,
              'route_50_invalid_event_time': invalid_time,
              'sep13_status_counts': dict(sorted(day_status.items())),
              'sep13_status_transaction_type_counts': [
                  {'status': s, 'transaction_type': t, 'rows': n}
                  for (s, t), n in sorted(types.items())],
              'sep13_hourly': sep13,
              'weekend_daily': [
                  {'date': day, 'raw_success': sum(raw[day, h] for h in range(24)),
                   'label': sum(labels[day, h] for h in range(24))}
                  for day in days if date.fromisoformat(day).weekday() >= 5]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('rows_scanned:', read_rows)
    print('sep13_status_counts:', report['sep13_status_counts'])
    print('sep13_label:', sum(x['label'] for x in sep13))
    print('sep13_raw_success:', sum(x['raw_success'] for x in sep13))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
