"""Track B (düzgün) — ETAS modelini fit eder ve baseline'lara karşı değerlendirir.

ETAS, train döneminde MLE ile fit edilir; sonra test 30g pencerelerinde beklenen
olay sayısını (yalnızca pencere öncesi geçmişten) öngörür. Climatology ve
gradient-boosting baseline'larıyla aynı out-of-time metrikte (Poisson deviance)
karşılaştırılır.

Kullanım:
    ./.venv/bin/python scripts/08_etas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness  # noqa: E402
from eyequake.config import PROCESSED_DIR, TURKEY  # noqa: E402
from eyequake.forecast.etas import branching_ratio, evaluate_etas_windows  # noqa: E402
from eyequake.forecast.evaluate import score, skill_score  # noqa: E402

MIN_MAG = 4.0
WINDOW_DAYS = 30.0
TRAIN_FRAC = 0.75


def main() -> int:
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])
    sub = df[df["mag"] >= MIN_MAG].copy()
    sub["time"] = pd.to_datetime(sub["time"], utc=True, format="ISO8601")
    sub = sub.sort_values("time").reset_index(drop=True)

    t0 = sub["time"].iloc[0]
    t = ((sub["time"] - t0).dt.total_seconds() / 86_400.0).to_numpy()  # gün
    m = sub["mag"].to_numpy()
    T_end = float(t[-1])

    ev = evaluate_etas_windows(t, m, m0=MIN_MAG, train_frac=TRAIN_FRAC, window_days=WINDOW_DAYS)
    params, k, T_split = ev.params, ev.n_train_events, ev.t_split

    print(f"M≥{MIN_MAG} | olay: {len(t)} | süre: {T_end / 365.25:.1f} yıl")
    print(f"Train: [0, {T_split / 365.25:.1f}yıl] ({k} olay) | Test: kalan")

    # --- ETAS fit (train) ---
    bval = b_value_aki(m, magnitude_of_completeness(m)).b
    n_branch = branching_ratio(params, bval)
    print("\n=== FİT EDİLEN ETAS PARAMETRELERİ ===")
    print(f"μ (background) = {params.mu:.4f} olay/gün")
    print(f"K = {params.K:.4f} | α = {params.alpha:.3f} | c = {params.c:.4f}g | p = {params.p:.3f}")
    print(f"log-olabilirlik = {params.loglik:.1f}")
    print(f"dallanma oranı n = {n_branch:.3f}" if n_branch else "dallanma oranı: tanımsız (α≥β)")

    # --- Test pencereleri vs baseline'lar (train pencere ortalaması + persistence) ---
    actual, etas_pred = ev.actual, ev.etas_pred
    clim, persistence = ev.climatology, ev.persistence

    s_etas = score(actual, etas_pred)
    s_clim = score(actual, clim)
    s_pers = score(actual, persistence)
    base = s_clim["poisson_deviance"]

    print(f"\n=== TEST ({len(actual)} pencere, {WINDOW_DAYS:.0f}g) ===")
    print(f"{'model':<22} {'MAE':>7} {'PoisDev':>9} {'beceri':>9}")
    for name, s in [("ETAS", s_etas), ("climatology", s_clim), ("persistence", s_pers)]:
        print(f"{name:<22} {s['mae']:>7.2f} {s['poisson_deviance']:>9.3f} "
              f"{skill_score(s['poisson_deviance'], base):>+9.1%}")

    sk = ev.skill_vs_climatology
    if sk > 0:
        print(f"\nVERDICT: ETAS climatology'yi %{sk * 100:.1f} yendi → fiziksel-temelli "
              "olasılıksal oran öngörüsü çalışıyor (artçı tetiklenmesi).")
    else:
        print("\nVERDICT: ETAS climatology'yi yenemedi — parametre/dönem incelenmeli.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
