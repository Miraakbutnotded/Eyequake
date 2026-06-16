"""Katalogu sabit zaman pencerelerine bölme ve denetimli (supervised) veri kurma.

Deprem öngörüsünü zaman-serisi problemine indirger: geçmiş N pencerenin
sismisitesinden bir sonraki pencereyi tahmin et. Zaman sırası KORUNUR — karıştırma
(shuffle) yapılmaz; aksi halde gelecekten geçmişe veri sızar (leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Supervised:
    """Lag-özellikli denetimli veri seti + pencere zaman damgaları."""

    X: np.ndarray
    y: np.ndarray
    window_starts: pd.DatetimeIndex
    feature_names: list[str]


def build_windowed_series(
    df: pd.DataFrame, freq_days: int = 30, min_mag: float = 4.0
) -> pd.DataFrame:
    """Olayları sabit pencerelere toplar.

    Returns:
        Pencere başlangıcına göre indeksli DataFrame:
          - n_events: penceredeki M≥min_mag olay sayısı
          - max_mag:  penceredeki en büyük büyüklük (olay yoksa 0.0)
    """
    sub = df[df["mag"] >= min_mag].copy()
    if not pd.api.types.is_datetime64_any_dtype(sub["time"]):
        sub["time"] = pd.to_datetime(sub["time"], utc=True, format="ISO8601")
    sub = sub.set_index("time").sort_index()

    rule = f"{freq_days}D"
    n_events = sub["mag"].resample(rule).size().rename("n_events")
    max_mag = sub["mag"].resample(rule).max().rename("max_mag").fillna(0.0)

    out = pd.concat([n_events, max_mag], axis=1)
    out["n_events"] = out["n_events"].fillna(0).astype(int)
    return out


def make_supervised(
    series: pd.DataFrame, n_lags: int = 6, target: str = "n_events"
) -> Supervised:
    """Geçmiş n_lags pencereden (n_events + max_mag) bir sonraki hedefi kurar.

    X satırı t: [n_events_{t-1..t-n_lags}, max_mag_{t-1..t-n_lags}]
    y satırı t: series[target]_t
    """
    n_events = series["n_events"].to_numpy(dtype=float)
    max_mag = series["max_mag"].to_numpy(dtype=float)
    y_all = series[target].to_numpy(dtype=float)
    starts = series.index

    rows, targets, kept_starts = [], [], []
    for t in range(n_lags, len(series)):
        feat = np.concatenate(
            [n_events[t - n_lags : t][::-1], max_mag[t - n_lags : t][::-1]]
        )
        rows.append(feat)
        targets.append(y_all[t])
        kept_starts.append(starts[t])

    names = [f"n_events_lag{i}" for i in range(1, n_lags + 1)]
    names += [f"max_mag_lag{i}" for i in range(1, n_lags + 1)]

    return Supervised(
        X=np.asarray(rows),
        y=np.asarray(targets),
        window_starts=pd.DatetimeIndex(kept_starts),
        feature_names=names,
    )
