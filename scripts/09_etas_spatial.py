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

import pandas as pd

from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness
from eyequake.config import FAULT_ZONES
from eyequake.data.clean import load_catalog
from eyequake.forecast.backtest import InsufficientData, backtest_etas, to_days
from eyequake.forecast.etas import branching_ratio
from eyequake.forecast.evaluate import score, skill_score

MIN_MAG = 4.0
WINDOW_DAYS = 30.0
TRAIN_FRAC = 0.75
MIN_EVENTS = 150  # bölge fiti için minimum olay sayısı


def fit_and_eval(label: str, sub: pd.DataFrame) -> dict | None:
    sub = sub.sort_values("time").reset_index(drop=True)
    t = to_days(sub["time"])
    m = sub["mag"].to_numpy()

    try:
        bt = backtest_etas(t, m, m0=MIN_MAG, window_days=WINDOW_DAYS, train_frac=TRAIN_FRAC)
    except InsufficientData as e:
        print(f"\n{label}: {e}, atlanıyor.")
        return None

    params = bt.params
    try:
        bval = b_value_aki(m, magnitude_of_completeness(m)).b
    except ValueError:
        bval = 1.0
    n_branch = branching_ratio(params, bval)

    s_etas = score(bt.actual, bt.etas_pred)
    s_clim = score(bt.actual, bt.climatology)
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
    df = load_catalog()
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
