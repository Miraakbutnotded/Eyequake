"""Track B+ uzantısı — bölge-bazlı (fay sistemi) ETAS fiti.

FINDINGS.md, ulusal havuzlanmış (pooled) ETAS fitinin mekanik olarak bağımsız
fay sistemlerini (KAF/DAF/Ege/Van) tek modelde topladığını ve bunun dallanma
oranı n'i fiziksel olmayan bir bölgeye itebileceğini belirtiyordu. Bu script
her bölge için ayrı ETAS fit eder ve n'in bölge bazında (n<1, sönen kaskad)
olup olmadığını raporlar.

Kullanım:
    ./.venv/bin/python scripts/09_etas_spatial.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness  # noqa: E402
from eyequake.config import FAULT_ZONES, PROCESSED_DIR, TURKEY  # noqa: E402
from eyequake.forecast.etas import branching_ratio, expected_count, fit_etas  # noqa: E402
from eyequake.forecast.evaluate import score, skill_score  # noqa: E402

MIN_MAG = 4.0
WINDOW_DAYS = 30.0
TRAIN_FRAC = 0.75
MIN_EVENTS = 150  # bölge fiti için minimum olay sayısı


def fit_and_eval(label: str, sub: pd.DataFrame) -> dict | None:
    sub = sub.sort_values("time").reset_index(drop=True)
    t0 = sub["time"].iloc[0]
    t = ((sub["time"] - t0).dt.total_seconds() / 86_400.0).to_numpy()
    m = sub["mag"].to_numpy()
    T_end = float(t[-1])
    k = int(len(t) * TRAIN_FRAC)
    if k < 30 or len(t) - k < 8:
        print(f"\n{label}: yetersiz olay ({len(t)}), atlanıyor.")
        return None
    T_split = float(t[k])

    params = fit_etas(t[:k], m[:k], m0=MIN_MAG, T0=0.0, T1=T_split)
    try:
        bval = b_value_aki(m, magnitude_of_completeness(m)).b
    except ValueError:
        bval = 1.0
    n_branch = branching_ratio(params, bval)

    win_starts = np.arange(T_split, T_end - WINDOW_DAYS, WINDOW_DAYS)
    if len(win_starts) == 0:
        print(f"\n{label}: test penceresi yok, atlanıyor.")
        return None

    actual, etas_pred = [], []
    for ws in win_starts:
        we = ws + WINDOW_DAYS
        actual.append(int(np.sum((t >= ws) & (t < we))))
        hist = t < ws
        etas_pred.append(expected_count(params, t[hist], m[hist], ws, we))
    actual = np.array(actual, dtype=float)
    etas_pred = np.array(etas_pred, dtype=float)

    train_starts = np.arange(0.0, T_split - WINDOW_DAYS, WINDOW_DAYS)
    train_counts = [int(np.sum((t >= s) & (t < s + WINDOW_DAYS))) for s in train_starts]
    clim = np.full(len(actual), float(np.mean(train_counts)) if train_counts else 0.0)

    s_etas = score(actual, etas_pred)
    s_clim = score(actual, clim)
    sk = (
        skill_score(s_etas["poisson_deviance"], s_clim["poisson_deviance"])
        if s_clim["poisson_deviance"]
        else 0.0
    )

    print(f"\n=== {label} ===")
    print(f"olay: {len(t)} (M≥{MIN_MAG}) | b-değeri: {bval:.3f}")
    print(f"μ={params.mu:.4f} K={params.K:.4f} α={params.alpha:.3f} c={params.c:.4f}g p={params.p:.3f}")
    print(f"dallanma oranı n = {n_branch:.3f}" if n_branch is not None else "dallanma oranı: tanımsız (α≥β)")
    print(
        f"ETAS vs climatology: MAE {s_etas['mae']:.2f} vs {s_clim['mae']:.2f} | "
        f"PoisDev {s_etas['poisson_deviance']:.2f} vs {s_clim['poisson_deviance']:.2f} | beceri {sk:+.1%}"
    )
    return {"label": label, "n_events": len(t), "n_branch": n_branch, "skill": sk}


def main() -> int:
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df[df["mag"] >= MIN_MAG].copy()

    print("=== ULUSAL HAVUZ (referans — FINDINGS.md'deki pooled fit) ===")
    national = fit_and_eval("Ulusal (havuzlanmış)", df)

    results = []
    for zone in FAULT_ZONES:
        sub = df[
            (df["latitude"] >= zone.min_lat)
            & (df["latitude"] <= zone.max_lat)
            & (df["longitude"] >= zone.min_lon)
            & (df["longitude"] <= zone.max_lon)
        ]
        if len(sub) < MIN_EVENTS:
            print(f"\n{zone.name}: yetersiz olay ({len(sub)} < {MIN_EVENTS}), atlanıyor.")
            continue
        r = fit_and_eval(zone.name, sub)
        if r:
            results.append(r)

    print("\n=== ÖZET: bölge-bazlı dallanma oranları ===")
    if national:
        results_with_national = [national] + results
    else:
        results_with_national = results
    for r in results_with_national:
        if r["n_branch"] is not None:
            flag = "fiziksel (n<1)" if r["n_branch"] < 1 else "n>=1 — sönmeyen kaskad"
            print(f"{r['label']:<28} olay={r['n_events']:>5}  n={r['n_branch']:.3f}  beceri={r['skill']:+.1%}  [{flag}]")
        else:
            print(f"{r['label']:<28} olay={r['n_events']:>5}  n=tanımsız  beceri={r['skill']:+.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
