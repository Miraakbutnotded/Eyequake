"""Track B (düzgün) — ETAS modelini fit eder ve baseline'lara karşı değerlendirir.

ETAS, train döneminde MLE ile fit edilir; sonra test 30g pencerelerinde beklenen
olay sayısını (yalnızca pencere öncesi geçmişten) öngörür. Climatology ve
gradient-boosting baseline'larıyla aynı out-of-time metrikte (Poisson deviance)
karşılaştırılır.

Kullanım:
    ./.venv/bin/python scripts/08_etas.py
"""

from __future__ import annotations

from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness
from eyequake.data.clean import load_catalog
from eyequake.forecast.backtest import backtest_etas, to_days
from eyequake.forecast.etas import branching_ratio
from eyequake.forecast.evaluate import score, skill_score

MIN_MAG = 4.0
WINDOW_DAYS = 30.0
TRAIN_FRAC = 0.75


def main() -> int:
    df = load_catalog()
    sub = df[df["mag"] >= MIN_MAG].sort_values("time").reset_index(drop=True)
    t = to_days(sub["time"])
    m = sub["mag"].to_numpy()

    print(f"M≥{MIN_MAG} | olay: {len(t)} | süre: {t[-1] / 365.25:.1f} yıl")

    bt = backtest_etas(t, m, m0=MIN_MAG, window_days=WINDOW_DAYS, train_frac=TRAIN_FRAC)
    params = bt.params
    print(f"Train: [0, {bt.t_split / 365.25:.1f}yıl] ({bt.n_train} olay) | Test: kalan")

    bval = b_value_aki(m, magnitude_of_completeness(m)).b
    n_branch = branching_ratio(params, bval)
    print("\n=== FİT EDİLEN ETAS PARAMETRELERİ ===")
    print(f"μ (background) = {params.mu:.4f} olay/gün")
    print(f"K = {params.K:.4f} | α = {params.alpha:.3f} | c = {params.c:.4f}g | p = {params.p:.3f}")
    print(f"log-olabilirlik = {params.loglik:.1f}")
    print(f"dallanma oranı n = {n_branch:.3f}" if n_branch else "dallanma oranı: tanımsız (α≥β)")

    s_etas = score(bt.actual, bt.etas_pred)
    s_clim = score(bt.actual, bt.climatology)
    s_pers = score(bt.actual, bt.persistence)
    base = s_clim["poisson_deviance"]

    print(f"\n=== TEST ({len(bt.actual)} pencere, {WINDOW_DAYS:.0f}g) ===")
    print(f"{'model':<22} {'MAE':>7} {'PoisDev':>9} {'beceri':>9}")
    for name, s in [("ETAS", s_etas), ("climatology", s_clim), ("persistence", s_pers)]:
        print(f"{name:<22} {s['mae']:>7.2f} {s['poisson_deviance']:>9.3f} "
              f"{skill_score(s['poisson_deviance'], base):>+9.1%}")

    sk = skill_score(s_etas["poisson_deviance"], base)
    if sk > 0:
        print(f"\nVERDICT: ETAS climatology'yi %{sk * 100:.1f} yendi → fiziksel-temelli "
              "olasılıksal oran öngörüsü çalışıyor (artçı tetiklenmesi).")
    else:
        print("\nVERDICT: ETAS climatology'yi yenemedi — parametre/dönem incelenmeli.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
