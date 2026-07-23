"""Track B/C ilk dürüst test: ML, baseline'ları yenebiliyor mu?

30 günlük pencerelerde, M≥4.0 olay sayısını tahmin eder. ML modellerini
(Poisson GLM, gradient boosting) climatology + persistence baseline'larına karşı
out-of-time test setinde ölçer. Beceri skoru >0 değilse model anlamsızdır.

Kullanım:
    ./.venv/bin/python scripts/03_forecast_baselines.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

from eyequake.data.clean import load_catalog
from eyequake.forecast.baselines import (
    climatology_forecast,
    persistence_forecast,
)
from eyequake.forecast.evaluate import score, skill_score, temporal_split
from eyequake.forecast.windows import build_windowed_series, make_supervised

FREQ_DAYS = 30
MIN_MAG = 4.0
N_LAGS = 6
TEST_FRAC = 0.25


def main() -> int:
    df = load_catalog()

    series = build_windowed_series(df, freq_days=FREQ_DAYS, min_mag=MIN_MAG)
    sup = make_supervised(series, n_lags=N_LAGS, target="n_events")
    split = temporal_split(sup.X, sup.y, test_frac=TEST_FRAC)

    print(f"Pencere: {FREQ_DAYS}g | M≥{MIN_MAG} | lag={N_LAGS}")
    print(f"Toplam pencere: {len(series)} | örnek: {len(sup.y)} "
          f"(eğitim {split.split_index}, test {len(sup.y) - split.split_index})")
    print(f"Pencere başına olay: ort={series['n_events'].mean():.1f} "
          f"medyan={series['n_events'].median():.0f} maks={series['n_events'].max()}")

    results: dict[str, dict[str, float]] = {}

    # --- Baseline'lar ---
    clim = climatology_forecast(split.y_train, len(split.y_test))
    results["climatology (Poisson)"] = score(split.y_test, clim)
    pers = persistence_forecast(split.y_train, split.y_test)
    results["persistence"] = score(split.y_test, pers)

    # --- ML modelleri ---
    glm = PoissonRegressor(alpha=1e-3, max_iter=500)
    glm.fit(split.X_train, split.y_train)
    results["poisson GLM"] = score(split.y_test, np.clip(glm.predict(split.X_test), 0, None))

    gbr = HistGradientBoostingRegressor(loss="poisson", max_depth=3, random_state=0)
    gbr.fit(split.X_train, split.y_train)
    results["gradient boosting"] = score(split.y_test, np.clip(gbr.predict(split.X_test), 0, None))

    # --- Rapor ---
    base = results["climatology (Poisson)"]["poisson_deviance"]
    print("\n=== TEST SETİ (out-of-time) ===")
    print(f"{'model':<24} {'MAE':>7} {'RMSE':>7} {'PoisDev':>9} {'beceri*':>9}")
    for name, m in results.items():
        sk = skill_score(m["poisson_deviance"], base)
        print(f"{name:<24} {m['mae']:>7.2f} {m['rmse']:>7.2f} "
              f"{m['poisson_deviance']:>9.3f} {sk:>+9.2%}")
    print("\n*beceri = climatology'ye göre Poisson deviance iyileşmesi (>0 = yener)")

    winner = min(results, key=lambda k: results[k]["poisson_deviance"])
    print(f"\nEn iyi: {winner}")
    if winner.startswith("climatology"):
        print("VERDICT: Hiçbir model background rate'i yenemedi → kısa-vade sinyali yok.")
    else:
        sk = skill_score(results[winner]["poisson_deviance"], base)
        print(f"VERDICT: {winner} climatology'yi %{sk * 100:.1f} yendi "
              "→ kümelenme (clustering) kaynaklı GERÇEK kısa-vade sinyali var.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
