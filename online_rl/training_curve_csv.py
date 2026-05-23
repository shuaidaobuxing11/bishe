"""
训练曲线 CSV 读取（TrainingCurveEvalCallback 输出）。
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

from online_rl.training_curve_callback import _migrate_csv_to_full


def parse_csv_float(cell: str) -> float:
    s = (cell or "").strip()
    if s == "":
        return float("nan")
    return float(s)


def read_curves_full(csv_path: str | Path) -> Dict[str, np.ndarray]:
    """
    method -> ndarray [training_steps, success_rate, mean_return, mean_length, collision_rate]
    同一时间步多条记录保留最后一次。
    """
    path = str(csv_path)
    _migrate_csv_to_full(path)

    buckets: dict[str, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_extra = "mean_length" in fieldnames and "collision_rate" in fieldnames
        for row in reader:
            method = (row.get("method") or "").strip()
            if not method:
                continue
            ts = int(float(row.get("training_steps", "0")))
            succ = parse_csv_float(row.get("success_rate", "0"))
            ret = parse_csv_float(row.get("mean_return", "0"))
            if has_extra:
                ml = parse_csv_float(row.get("mean_length", ""))
                cr = parse_csv_float(row.get("collision_rate", ""))
            else:
                ml = float("nan")
                cr = float("nan")
            buckets[method][ts] = (succ, ret, ml, cr)

    out: Dict[str, np.ndarray] = {}
    for method, by_ts in buckets.items():
        ordered = sorted(by_ts.items(), key=lambda x: x[0])
        rows = [[ts] + list(v) for ts, v in ordered]
        out[method] = np.asarray(rows, dtype=np.float64)
    return out
