#!/usr/bin/env python3
"""Потоковый аудит валидаций. Пишет только агрегаты, без строк транзакций.

Python 3.10+, только стандартная библиотека. Не загружает CSV целиком в память.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date
from pathlib import Path

ROUTES = {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
RAW_COLUMNS = {"tran_date_time", "validation_result", "ngpt_route"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@contextmanager
def open_text(path: str):
    """Путь CSV, CSV.gz или archive.zip::member.csv."""
    if "::" in path:
        archive, member = path.split("::", 1)
        with zipfile.ZipFile(archive) as z:
            with z.open(member) as binary:
                import io
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    yield text
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as text:
            yield text
    else:
        with open(path, encoding="utf-8-sig", newline="") as text:
            yield text


def parse_timestamp(value: str, date_cache: dict[str, bool]):
    # Берём время события, не время загрузки записи. Без предположения о TZ.
    day = value[:10]
    if len(value) < 13 or not DATE_RE.fullmatch(day) or value[10] not in " T":
        return None
    if day not in date_cache:
        try:
            date.fromisoformat(day)
            date_cache[day] = True
        except ValueError:
            date_cache[day] = False
    if not date_cache[day]:
        return None
    try:
        hour = int(value[11:13])
    except ValueError:
        return None
    if not 0 <= hour <= 23 or len(value) < 16 or value[13] != ":":
        return None
    return day, hour


def parse_route(value: str):
    match = re.match(r"^\s*(\d+)\b", value or "")
    return int(match.group(1)) if match else None


def write_csv(path: Path, columns: list[str], rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(columns)
        writer.writerows(rows)


def load_labels(paths: list[str]):
    labels = {}
    report = {}
    for path in paths:
        count = 0
        duplicates = 0
        route_count = Counter()
        with open_text(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            if not {"route", "date", "hour", "boardings"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Неверная схема labels: {path}")
            for row in reader:
                key = (int(row["route"]), row["date"], int(row["hour"]))
                if key in labels:
                    duplicates += 1
                labels[key] = int(row["boardings"])
                route_count[key[0]] += 1
                count += 1
        report[Path(path).name] = {"rows": count, "duplicate_keys": duplicates,
                                   "routes": dict(sorted(route_count.items()))}
    return labels, report


def inspect_submission(path: str):
    keys = set()
    duplicate = 0
    bad = 0
    per_route = Counter()
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames != ["route", "date", "hour", "prediction"]:
            raise ValueError(f"Неверные колонки submission: {reader.fieldnames}")
        for row in reader:
            try:
                route, hour = int(row["route"]), int(row["hour"])
                day = date.fromisoformat(row["date"])
                prediction = float(row["prediction"])
                if route not in ROUTES or not 0 <= hour < 24 or not math.isfinite(prediction) or prediction < 0:
                    bad += 1
                key = (route, day.isoformat(), hour)
                duplicate += key in keys
                keys.add(key)
                per_route[route] += 1
            except (ValueError, OverflowError):
                bad += 1
    return {"rows": sum(per_route.values()), "unique_keys": len(keys),
            "duplicate_keys": duplicate, "invalid_rows": bad,
            "routes": dict(sorted(per_route.items())),
            "expected_2025_nov_dec_rows": 14640,
            "exact_expected_grid": keys == {(r, d.isoformat(), h)
               for r in ROUTES for d in days(date(2025, 11, 1), date(2025, 12, 31))
               for h in range(24)}}


def days(start: date, end: date):
    from datetime import timedelta
    while start <= end:
        yield start
        start += timedelta(days=1)


def audit_raw(path: str, max_rows: int, progress_every: int):
    hourly = Counter()
    statuses = Counter()
    transaction_types = Counter()
    by_route_status = Counter()
    row_count = 0
    valid_timestamp = 0
    invalid_timestamp = 0
    unknown_route = 0
    blank_route = 0
    date_cache: dict[str, bool] = {}
    start = time.monotonic()
    with open_text(path) as f:
        reader = csv.DictReader(f, delimiter=";")
        if not RAW_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"Не найдены обязательные колонки: {RAW_COLUMNS - set(reader.fieldnames or [])}")
        for row in reader:
            row_count += 1
            route = parse_route(row["ngpt_route"])
            stamp = parse_timestamp(row["tran_date_time"], date_cache)
            status = row["validation_result"]
            statuses[status] += 1
            transaction_types[row.get("tran_type_id", "")] += 1
            if route is None:
                blank_route += 1
            elif route not in ROUTES:
                unknown_route += 1
            if stamp is None:
                invalid_timestamp += 1
            else:
                valid_timestamp += 1
                if route in ROUTES:
                    by_route_status[(route, status)] += 1
                    if status == "1":
                        hourly[(route, stamp[0], stamp[1])] += 1
            if progress_every and row_count % progress_every == 0:
                print(f"rows={row_count:,} elapsed_s={time.monotonic()-start:.0f}",
                      file=sys.stderr, flush=True)
            if max_rows and row_count >= max_rows:
                break
    return hourly, {
        "raw_rows_read": row_count, "partial_scan": bool(max_rows),
        "valid_tran_date_time": valid_timestamp,
        "invalid_tran_date_time": invalid_timestamp,
        "missing_or_unparseable_route": blank_route,
        "route_outside_submission": unknown_route,
        "validation_result_counts": dict(statuses.most_common()),
        "tran_type_id_counts": dict(transaction_types.most_common()),
        "route_validation_result_counts": [
            {"route": r, "validation_result": s, "count": n}
            for (r, s), n in sorted(by_route_status.items())],
        "distinct_dates": len(date_cache), "elapsed_seconds": round(time.monotonic()-start, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="train.csv, train.csv.gz или test.zip::test.csv")
    p.add_argument("--labels", action="append", default=[], help="Повторять для train и test")
    p.add_argument("--submission", help="test_submission.csv")
    p.add_argument("--out", required=True, type=Path, help="Папка ТОЛЬКО с агрегатами")
    p.add_argument("--max-rows", type=int, default=0, help="Быстрая проба; 0 = весь файл")
    p.add_argument("--progress-every", type=int, default=2_000_000)
    a = p.parse_args()
    if not any((a.input, a.labels, a.submission)):
        p.error("Нужен --input, --labels или --submission")
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "note": "Только агрегаты; исходные транзакции и хеши не сохраняются"}
    labels, label_report = load_labels(a.labels)
    if a.labels:
        report["labels"] = label_report
        report["labels_total_boardings"] = sum(labels.values())
    if a.submission:
        report["submission"] = inspect_submission(a.submission)
    if a.input:
        hourly, raw = audit_raw(a.input, a.max_rows, a.progress_every)
        report["raw"] = raw
        write_csv(a.out / "hourly_route.csv", ["route", "date", "hour", "boardings"],
                  ((*key, value) for key, value in sorted(hourly.items())))
        daily = Counter()
        for (route, day, _), n in hourly.items():
            daily[(route, day)] += n
        write_csv(a.out / "daily_route.csv", ["route", "date", "boardings"],
                  ((*key, value) for key, value in sorted(daily.items())))
        if labels and not a.max_rows:
            keys = set(hourly) | set(labels)
            mismatched = [k for k in keys if hourly.get(k, 0) != labels.get(k, 0)]
            by_date = Counter(k[1] for k in mismatched)
            report["reconciliation"] = {
                "compared_union_keys": len(keys), "mismatched_keys": len(mismatched),
                "sum_raw": sum(hourly.values()), "sum_labels": sum(labels.values()),
                "mismatches_by_date": dict(sorted(by_date.items())),
                "warning": "Края файлов могут пересекаться: первые часы сентября в хвосте train, первые часы ноября в хвосте test. Сверяйте после объединения и фильтра по периоду.",
            }
    (a.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {a.out / 'summary.json'}")
    print("Можно прислать summary.json; hourly_route.csv и daily_route.csv содержат только агрегаты.")


if __name__ == "__main__":
    main()
