"""Models receive ONLY history through cutoff. Predict never accepts future truth."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from statistics import median, mean

from .core import grid, rounded


class Profile:
    def __init__(self, statistic='mean', missing='zero', shrink=0.2, window=None):
        if statistic not in ('mean', 'median') or missing not in ('zero', 'observed'):
            raise ValueError('Invalid profile configuration')
        self.statistic, self.missing, self.shrink, self.window = statistic, missing, shrink, window

    def fit(self, history, start, cutoff):
        if not history or any(not start <= k[1] <= cutoff for k in history):
            raise ValueError('Training history crosses cutoff or is empty')
        self.cutoff = cutoff
        self.routes = sorted({k[0] for k in history})
        if self.window:
            start = max(start, cutoff - timedelta(days=self.window - 1))
        history = {k: v for k, v in history.items() if k[1] >= start}
        local, hour, route = defaultdict(list), defaultdict(list), defaultdict(list)
        keys = grid(self.routes, start, cutoff) if self.missing == 'zero' else history
        for r, d, h in keys:
            y = history.get((r, d, h), 0)
            local[r, d.weekday(), h].append(y)
            hour[r, h].append(y)
            route[r].append(y)
        fn = mean if self.statistic == 'mean' else median
        self.local = {k: fn(v) for k, v in local.items()}
        self.hour = {k: fn(v) for k, v in hour.items()}
        self.route = {k: fn(v) for k, v in route.items()}
        return self

    def predict(self, keys):
        if any(k[1] <= self.cutoff for k in keys):
            raise ValueError('Prediction dates must be after cutoff')
        result = {}
        for key in keys:
            r, d, h = key
            # Completely unseen routes have no empirical estimate. Explicit zero cold start.
            parent = self.hour.get((r, h), self.route.get(r, 0))
            local = self.local.get((r, d.weekday(), h), parent)
            result[key] = rounded((1 - self.shrink) * local + self.shrink * parent)
        return result


class Boosting:
    def __init__(self, recursive=False):
        self.recursive = recursive

    def features(self, keys, context):
        import numpy as np
        result = []
        for r, d, h in keys:
            t = d.timetuple().tm_yday
            row = [self.route_codes[r], d.weekday(), h, d.month, t,
                   int(d.weekday() >= 5), math.sin(2 * math.pi * t / 365),
                   math.cos(2 * math.pi * t / 365)]
            if self.recursive:
                for lag in (1, 7):
                    key = r, d - timedelta(days=lag), h
                    row.extend([context.get(key, 0), int(key in context)])
            result.append(row)
        return np.asarray(result, dtype=float)

    def fit(self, history, start, cutoff):
        from sklearn.ensemble import HistGradientBoostingRegressor
        if not history or any(not start <= k[1] <= cutoff for k in history):
            raise ValueError('Training history crosses cutoff or is empty')
        self.cutoff = cutoff
        self.routes = sorted({k[0] for k in history})
        self.route_codes = {r: i for i, r in enumerate(self.routes)}
        # Complete training grid: zero-fill assumption is reported separately from facts.
        keys = grid(self.routes, start, cutoff)
        if self.recursive:
            keys = [k for k in keys if k[1] >= start + timedelta(days=7)]
        self.history = dict(history)
        self.model = HistGradientBoostingRegressor(
            loss='absolute_error', learning_rate=.08, max_iter=150,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2,
            categorical_features=[0, 1, 2, 3, 5], early_stopping=False, random_state=2025)
        self.model.fit(self.features(keys, history), [history.get(k, 0) for k in keys])
        return self

    def predict(self, keys):
        if any(k[1] <= self.cutoff for k in keys):
            raise ValueError('Prediction dates must be after cutoff')
        known = [k for k in keys if k[0] in self.route_codes]
        result = {k: 0 for k in keys}
        if not known:
            return result
        if not self.recursive:
            result.update(zip(known, map(rounded, self.model.predict(self.features(known, {})))))
            return result
        # Generate every intervening day. No observed validation targets ever enter context.
        context = dict(self.history)
        day, end = self.cutoff + timedelta(days=1), max(k[1] for k in known)
        wanted = set(known)
        while day <= end:
            batch = grid(self.routes, day, day)
            predictions = self.model.predict(self.features(batch, context))
            for key, value in zip(batch, predictions):
                context[key] = rounded(value)
                if key in wanted:
                    result[key] = context[key]
            day += timedelta(days=1)
        return result


def candidates(include_boosting=True):
    result = {
        'mean_zero': lambda: Profile('mean'),
        'median_zero': lambda: Profile('median'),
        'mean_observed': lambda: Profile('mean', 'observed'),
        'median_observed': lambda: Profile('median', 'observed'),
        'mean_56d': lambda: Profile('mean', window=56),
    }
    if include_boosting:
        result.update(hgb_direct=Boosting, hgb_recursive=lambda: Boosting(recursive=True))
    return result
