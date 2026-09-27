#!/usr/bin/env python3
"""Ablation: меняется только длина известной истории, валидация неизменна.

Кандидаты fit/evaluate из backtest_baseline.py. Во время предсказания всего
среза истинные значения из него не используются. JSON содержит лишь агрегаты.
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from backtest_baseline import evaluate, load


FOLDS = {
    'jul_aug': (date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
    'sep_oct': (date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
}
WINDOWS = (None, 28, 56, 84, 112, 168)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    data = load(args.train) | load(args.test)
    routes = sorted({r for r, _, _ in data})
    first = date(2025, 1, 1)
    report = {'hypothesis': 'Recent history improves a route-weekday-hour forecast',
              'changed_parameter': 'fit history length; model, label rule, route grid and folds fixed',
              'leakage_rule': 'entire validation interval predicted at once; cutoff strictly before it',
              'routes': routes, 'folds': {}}
    for fold, (cutoff, start, end) in FOLDS.items():
        fold_report = {'cutoff': cutoff.isoformat(),
                       'validation': [start.isoformat(), end.isoformat()], 'variants': {}}
        training = {k: v for k, v in data.items() if first <= k[1] <= cutoff}
        truth = {k: v for k, v in data.items() if start <= k[1] <= end}
        for window in WINDOWS:
            fit_start = max(first, cutoff - timedelta(days=window - 1)) if window else first
            fit_data = {k: v for k, v in training.items() if k[1] >= fit_start}
            for statistic in ('mean', 'median'):
                key = f'{statistic}_' + (str(window) + 'd' if window else 'all')
                metrics, _ = evaluate(fit_data, truth, routes, fit_start, cutoff,
                                      start, end, statistic)
                fold_report['variants'][key] = {
                    'train_from': fit_start.isoformat(),
                    'metrics': {'all': metrics['all'],
                                'by_route_score': {r: m['wape_score']
                                                   for r, m in metrics.items() if r != 'all'}},
                }
        report['folds'][fold] = fold_report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    for fold, info in report['folds'].items():
        print(fold, sorted([(v['metrics']['all']['wape_score'], k)
                            for k, v in info['variants'].items()], reverse=True)[:4])
    print('Готово:', args.out)


if __name__ == '__main__':
    main()
