"""ETAS geriye-dönük test (backtest) çekirdeği — 08/09 scriptlerinin ortak mantığı.

Akış her iki scriptte de aynıdır: katalogu güne çevir → train/test'i zamanda böl →
train'de ETAS fit et → test penceresi başına beklenen sayıyı üret → gerçek sayımla
ve baseline'larla karşılaştır. Bu mantık scriptlerde kopyalanmış hâldeyken test
edilemiyordu; buraya taşındı.

Değişmez: her pencere tahmini YALNIZCA pencere başından önceki olayları görür
(out-of-time, sızıntısız).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .etas import ETASParams, expected_count, fit_etas


class InsufficientData(ValueError):
    """Fit ya da test penceresi için yeterli olay yok."""


def to_days(times: pd.Series) -> np.ndarray:
    """Zaman damgalarını ilk olaydan itibaren geçen güne çevirir."""
    times = pd.to_datetime(times, utc=True, format="ISO8601")
    return ((times - times.iloc[0]).dt.total_seconds() / 86_400.0).to_numpy()


def window_starts(t_from: float, t_to: float, window_days: float) -> np.ndarray:
    """[t_from, t_to) aralığındaki pencere başlangıçları (son taşan pencere hariç)."""
    return np.arange(t_from, t_to - window_days, window_days)


def window_counts(t: np.ndarray, starts: np.ndarray, window_days: float) -> np.ndarray:
    """Her pencerede gerçekleşen olay sayısı."""
    return np.array(
        [np.sum((t >= s) & (t < s + window_days)) for s in starts], dtype=float
    )


def etas_forecast(
    params: ETASParams, t: np.ndarray, m: np.ndarray,
    starts: np.ndarray, window_days: float,
) -> np.ndarray:
    """Pencere başına beklenen olay sayısı — yalnızca geçmiş olaylardan."""
    preds = []
    for s in starts:
        hist = t < s
        preds.append(expected_count(params, t[hist], m[hist], s, s + window_days))
    return np.array(preds, dtype=float)


@dataclass(frozen=True)
class Backtest:
    """Bir ETAS backtest'inin tüm ara ürünleri (skorlama çağıran tarafta)."""

    params: ETASParams
    starts: np.ndarray
    actual: np.ndarray
    etas_pred: np.ndarray
    climatology: np.ndarray
    persistence: np.ndarray
    n_events: int
    n_train: int
    t_split: float


def backtest_etas(
    t: np.ndarray, m: np.ndarray, m0: float,
    window_days: float = 30.0, train_frac: float = 0.75,
    min_train_events: int = 30, min_test_events: int = 8,
) -> Backtest:
    """Zaman-bölünmeli ETAS backtest'i: fit (train) → pencere tahmini (test).

    Baseline'lar aynı bölünmeden türetilir: climatology = train pencerelerinin
    ortalaması, persistence = bir önceki pencerenin gerçek sayımı.

    Raises:
        InsufficientData: fit ya da test için yeterli olay/pencere yoksa.
    """
    k = int(len(t) * train_frac)
    if k < min_train_events or len(t) - k < min_test_events:
        raise InsufficientData(f"yetersiz olay ({len(t)})")

    t_end = float(t[-1])
    t_split = float(t[k])
    params = fit_etas(t[:k], m[:k], m0=m0, T0=0.0, T1=t_split)

    starts = window_starts(t_split, t_end, window_days)
    if len(starts) == 0:
        raise InsufficientData("test penceresi yok")

    actual = window_counts(t, starts, window_days)
    etas_pred = etas_forecast(params, t, m, starts, window_days)

    train_counts = window_counts(t, window_starts(0.0, t_split, window_days), window_days)
    clim_level = float(np.mean(train_counts)) if train_counts.size else 0.0
    climatology = np.full(len(actual), clim_level)
    persistence = np.concatenate([[clim_level], actual[:-1]])

    return Backtest(
        params=params, starts=starts, actual=actual, etas_pred=etas_pred,
        climatology=climatology, persistence=persistence,
        n_events=len(t), n_train=k, t_split=t_split,
    )
