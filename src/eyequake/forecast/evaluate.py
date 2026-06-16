"""Zaman-serisi dürüst değerlendirme: temporal split + sayım metrikleri.

Karıştırma YOK. Test her zaman eğitimin gelecekindedir (out-of-time).
Poisson deviance, sayım (count) tahmininde MAE/RMSE'den daha bilgilendiricidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    split_index: int


def temporal_split(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25) -> Split:
    """Son test_frac oranını test olarak ayırır (zaman sırasını korur)."""
    n = len(y)
    k = int(round(n * (1.0 - test_frac)))
    return Split(X[:k], X[k:], y[:k], y[k:], k)


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ortalama Poisson sapması. Düşük = iyi. Tahmini >0'a kırpar."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    # 2 * (y*log(y/yhat) - (y - yhat)); y=0 için log terimi 0.
    term = np.where(yt > 0, yt * np.log(yt / yp), 0.0)
    return float(2.0 * np.mean(term - (yt - yp)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE ve Poisson deviance."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "poisson_deviance": poisson_deviance(yt, yp),
    }


def skill_score(model_metric: float, baseline_metric: float) -> float:
    """Beceri skoru: baseline'a göre iyileşme oranı. >0 ise model baseline'ı yener.

    1 - (model / baseline). 0 = baseline ile aynı, negatif = baseline'dan kötü.
    """
    if baseline_metric == 0:
        return 0.0
    return 1.0 - (model_metric / baseline_metric)
