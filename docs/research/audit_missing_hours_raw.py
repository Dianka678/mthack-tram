#!/usr/bin/env python3
"""Сверяет все пустые часы 06–22 в метках с сырыми записями всех статусов.

Не сохраняет идентификаторы пассажиров, карт, вагонов или строк транзакций.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from backtest_baseline import fit, load
from profile_data import open_text, parse_route, parse_timestamp


def days(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    present = set()
    for path in (a.train, a.test):
        with open(path, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f, delimiter=';'):
                present.add((int(row['route']), row['date'], int(row['hour'])))
    routes = sorted({r for r, _, _ in present})
    missing = {(r, day.isoformat(), hour)
               for day in days(date(2025, 9, 1), date(2025, 10, 31))
               for r in routes for hour in range(6, 23)
               if (r, day.isoformat(), hour) not in present}
    if {r for r, _, _ in missing} - {50}:
        raise ValueError('Missing daytime labels outside route 50; expand the raw audit before interpreting.')
    counts, types = Counter(), Counter()
    cache = {}
    total_raw = 0
    with open_text(a.raw) as f:
        reader = csv.DictReader(f, delimiter=';')
        need = {'ngpt_route', 'tran_date_time', 'validation_result', 'tran_type_id'}
        if not need.issubset(reader.fieldnames or []):
            raise ValueError(f'Raw fields missing: {need-set(reader.fieldnames or [])}')
        for row in reader:
            total_raw += 1
            if parse_route(row['ngpt_route']) != 50:
                continue
            stamp = parse_timestamp(row['tran_date_time'], cache)
            if not stamp:
                continue
            day, hour = stamp
            key = (50, day, hour)
            status = row['validation_result']
            tran_type = row['tran_type_id']
            if key in missing:
                counts[(day, hour, status)] += 1
                types[(status, tran_type)] += 1
    observed = [{'route': r, 'date': d, 'hour': h,
                 'raw_statuses': dict(sorted((status, n)
                      for (day, hr, status), n in counts.items() if (day, hr) == (d, h)))}
                for r, d, h in sorted(missing)]
    profile = fit(load(a.train), routes, date(2025, 1, 1), date(2025, 8, 31), 'median')
    local, by_hour, _ = profile
    oracle_removed_error = sum(round(.8 * local[r, date.fromisoformat(d).weekday(), h]
                                    + .2 * by_hour[r, h]) for r, d, h in missing)
    result = {
        'raw_rows_read': total_raw, 'scope': '2025-09-01 through 2025-10-31, 06:00-22:00',
        'candidate_hours_not_in_labels': len(missing),
        'candidate_hours_with_any_raw_transaction': sum(bool(x['raw_statuses']) for x in observed),
        'candidate_hours_with_success': sum('1' in x['raw_statuses'] for x in observed),
        'oracle_125_hour_error_removed_from_aug31_median': oracle_removed_error,
        'oracle_score_gain_if_future_missing_keys_known': round(
            oracle_removed_error / sum(load(a.test).values()), 6),
        'raw_status_totals': dict(sorted(Counter({s: sum(n for (_,_,st),n in counts.items()
                                          if st==s) for _,_,s in counts}).items())),
        'raw_transaction_types': [{'status': st, 'tran_type_id': typ, 'count': n}
                                  for (st, typ), n in sorted(types.items())],
        'missing_hours': observed,
        'limitations': 'No vehicle schedule or validator heartbeat; no raw transaction does not prove whether service ran.'
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('rows', total_raw, 'missing', len(missing), 'raw hits',
          result['candidate_hours_with_any_raw_transaction'], 'statuses', result['raw_status_totals'])


if __name__ == '__main__':
    main()
