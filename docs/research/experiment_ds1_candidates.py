#!/usr/bin/env python3
"""Оценка кандидатов PR DS-1 на реальных метках без изменения его ветки.

Использует `ml/forecast/core.py` и `models.py` из локального checkout
ветки ds1/forecast-submission. Историю строго ограничивает датой отсечения.
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

FOLDS = [
    ('may_jun', date(2025, 4, 30), date(2025, 5, 1), date(2025, 6, 30)),
    ('jul_aug', date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    ('sep_oct', date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
]
FIRST = date(2025, 1, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ds1-root', required=True, type=Path)
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--test', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--profiles-only', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(args.ds1_root.resolve()))
    from ml.forecast.core import grid, load_labels, peak_hours, score, sha256
    from ml.forecast.models import candidates

    values = load_labels((args.train, args.test))
    factories = candidates(not args.profiles_only)
    report = {
        'source_branch': 'ds1/forecast-submission',
        'source_commit': '613da6c8e5134107af4094db45f4433a0ced76e1',
        'ds1_code_sha256': {
            path: sha256(args.ds1_root / 'ml/forecast' / path)
            for path in ('core.py', 'models.py')},
        'input_sha256': {'train': sha256(args.train), 'test': sha256(args.test)},
        'metric': '1 - sum(abs(error))/sum(truth), clipped at zero',
        'policy': 'missing target row = zero for score; full hours; no validation truth in model features',
        'folds': {}, 'selection': {},
    }
    for fold, cutoff, start, end in FOLDS:
        if fold == 'sep_oct':
            dev = report['folds']
            losses = {name: sum(dev[f][name]['all']['absolute_error_sum']
                                for f in ('may_jun', 'jul_aug')) for name in factories}
            candidate = min(losses, key=lambda n: (losses[n], n))
            tested = list(dict.fromkeys((candidate, 'mean_zero', 'median_zero')))
            report['selection']['development_candidate'] = candidate
            report['selection']['development_absolute_errors'] = losses
        else:
            tested = list(factories)
        history = {k: v for k, v in values.items() if FIRST <= k[1] <= cutoff}
        truth = {k: v for k, v in values.items() if start <= k[1] <= end}
        routes = sorted({k[0] for k in history}, key=int)
        keys = grid(routes, start, end)
        report['folds'][fold] = {}
        for name in tested:
            t = time.perf_counter()
            model = factories[name]().fit(history, FIRST, cutoff)
            predictions = model.predict(keys)
            metrics = score(keys, truth, predictions, peak_hours(history, routes))
            report['folds'][fold][name] = {**metrics, 'elapsed_seconds': round(time.perf_counter() - t, 3)}
            print(f'{fold} {name}: {metrics["all"]["wape_score"]:.6f}', flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    last = report['folds']['sep_oct']
    baseline = min(('mean_zero', 'median_zero'), key=lambda n: last[n]['all']['absolute_error_sum'])
    candidate = report['selection']['development_candidate']
    report['selection']['validation_baseline'] = baseline
    report['selection']['selected'] = (candidate if last[candidate]['all']['absolute_error_sum']
                                      < last[baseline]['all']['absolute_error_sum'] else baseline)
    report['selection']['sep_oct_is_a_selection_gate_not_untouched_test'] = True
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('selected:', report['selection']['selected'])
    print('saved:', args.out)


if __name__ == '__main__':
    main()
