"""Track C dürüst test + ikili 'M≥5 gelecek pencerede' olasılıksal görev.

Bölüm 1 (Track C): Bir sonraki olayın büyüklüğü tahmin edilebilir mi?
  → ML'i 'ortalamayı tahmin et' baseline'ına karşı ölç. R²≈0 = tahmin edilemez.

Bölüm 2 (Track B uzantısı): Gelecek 30g penceresinde M≥5 olur mu? (olasılıksal)
  → Brier skor + ROC-AUC; base-rate'e karşı Brier beceri skoru.

Kullanım:
    ./.venv/bin/python scripts/05_deterministic_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score  # noqa: E402

from eyequake.config import PROCESSED_DIR, TURKEY  # noqa: E402
from eyequake.forecast.deterministic import build_event_features  # noqa: E402
from eyequake.forecast.evaluate import temporal_split  # noqa: E402
from eyequake.forecast.windows import build_windowed_series, make_supervised  # noqa: E402

MIN_MAG = 4.0
STRONG = 5.0


def track_c(df: pd.DataFrame) -> None:
    feats = build_event_features(df, min_mag=MIN_MAG, k=10)
    split = temporal_split(feats.X, feats.y, test_frac=0.25)

    clim = np.full(len(split.y_test), float(split.y_train.mean()))
    persistence = split.X_test[:, 0]  # last_mag özelliği
    gbr = HistGradientBoostingRegressor(max_depth=3, random_state=0)
    gbr.fit(split.X_train, split.y_train)
    ml = gbr.predict(split.X_test)

    print("=== TRACK C: bir sonraki olayın BÜYÜKLÜĞÜ ===")
    print(f"olay sayısı: {len(feats.y)} (test {len(split.y_test)})")
    print(f"{'model':<22} {'MAE':>7} {'R²':>8}")
    for name, pred in [("climatology (ort.)", clim),
                       ("persistence (son M)", persistence),
                       ("gradient boosting", ml)]:
        mae = mean_absolute_error(split.y_test, pred)
        r2 = r2_score(split.y_test, pred)
        print(f"{name:<22} {mae:>7.3f} {r2:>8.3f}")
    r2_ml = r2_score(split.y_test, ml)
    if r2_ml < 0.05:
        print(f"VERDICT: ML R²={r2_ml:.3f} ≈ 0 → bir sonraki büyüklük "
              "TAHMİN EDİLEMİYOR (Gutenberg-Richter beklentisiyle uyumlu).")
    else:
        print(f"VERDICT: ML R²={r2_ml:.3f} → beklenmedik sinyal, incele.")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def binary_task(df: pd.DataFrame) -> None:
    series = build_windowed_series(df, freq_days=30, min_mag=MIN_MAG)
    series = series.assign(strong=(series["max_mag"] >= STRONG).astype(int))
    sup = make_supervised(series, n_lags=6, target="strong")
    split = temporal_split(sup.X, sup.y, test_frac=0.25)

    base_rate = float(split.y_train.mean())
    base_pred = np.full(len(split.y_test), base_rate)

    clf = HistGradientBoostingClassifier(max_depth=3, random_state=0)
    clf.fit(split.X_train, split.y_train)
    ml_pred = clf.predict_proba(split.X_test)[:, 1]

    b_base = brier(split.y_test, base_pred)
    b_ml = brier(split.y_test, ml_pred)
    skill = 1.0 - b_ml / b_base if b_base else 0.0
    auc = roc_auc_score(split.y_test, ml_pred) if split.y_test.sum() else float("nan")

    print(f"\n=== İKİLİ: gelecek 30g penceresinde M≥{STRONG}? (olasılıksal) ===")
    print(f"test pozitif oran: {split.y_test.mean():.2%} (base rate {base_rate:.2%})")
    print(f"Brier  base-rate: {b_base:.4f} | ML: {b_ml:.4f} | beceri: {skill:+.1%}")
    print(f"ROC-AUC (ML): {auc:.3f}")
    if skill > 0 and auc > 0.55:
        print("VERDICT: M≥5 oluşumu için GERÇEK olasılıksal sinyal (kümelenme).")
    else:
        print("VERDICT: base-rate üstünde anlamlı sinyal yok.")


def main() -> int:
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])
    track_c(df)
    binary_task(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
