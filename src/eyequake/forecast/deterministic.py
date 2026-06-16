"""Track C — deterministik tahmin testi: bir sonraki olayın BÜYÜKLÜĞÜ.

Pazarlanan iddia: "şu büyüklükte deprem olacak". Dürüst test: geçmiş K olayın
istatistiğinden bir sonraki olayın büyüklüğünü tahmin etmeye çalış, baseline'a
(ortalamayı tahmin et) karşı ölç. Gutenberg-Richter, büyüklüklerin yaklaşık
bağımsız ve üstel dağıldığını söyler — bu yüzden R²≈0 beklenir. Veri konuşsun.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class EventFeatures:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    times: pd.DatetimeIndex


def _to_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, utc=True, format="ISO8601")


def build_event_features(
    df: pd.DataFrame, min_mag: float = 4.0, k: int = 10
) -> EventFeatures:
    """Her olay için, önceki k olaydan türetilen özellikler → hedef: o olayın M'i."""
    sub = df[df["mag"] >= min_mag].copy()
    sub["time"] = _to_utc(sub["time"])
    sub = sub.sort_values("time").reset_index(drop=True)

    mags = sub["mag"].to_numpy(dtype=float)
    times = sub["time"]
    t_sec = times.astype("int64").to_numpy() / 1e9  # epoch saniye

    rows, targets, kept = [], [], []
    for i in range(k, len(sub)):
        window = mags[i - k : i]
        span_days = max((t_sec[i - 1] - t_sec[i - k]) / _SECONDS_PER_DAY, 1e-6)
        feat = [
            mags[i - 1],                                   # son büyüklük
            float(window.mean()),                          # son k ortalama
            float(window.max()),                           # son k maks
            float(window.std()),                           # son k std
            (t_sec[i] - t_sec[i - 1]) / _SECONDS_PER_DAY,  # son olaydan beri gün
            k / span_days,                                 # son k oran (olay/gün)
        ]
        rows.append(feat)
        targets.append(mags[i])
        kept.append(times.iloc[i])

    names = [
        "last_mag", "recent_mean_mag", "recent_max_mag",
        "recent_std_mag", "days_since_last", "recent_rate_per_day",
    ]
    return EventFeatures(
        X=np.asarray(rows),
        y=np.asarray(targets),
        feature_names=names,
        times=pd.DatetimeIndex(kept),
    )
