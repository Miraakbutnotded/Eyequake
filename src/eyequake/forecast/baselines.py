"""Dürüst baseline'lar. Bir modelin anlamlı olması için bunları YENMESİ gerekir.

- Climatology / Poisson: eğitim ortalamasını sabit tahmin et. Deprem oranının
  hafızasız (memoryless) olduğu varsayımının karşılığı. Yenilmesi en zor olan.
- Persistence: bir sonraki pencere = önceki pencere. Otokorelasyon varsa güçlü.
"""

from __future__ import annotations

import numpy as np


def climatology_forecast(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Eğitim ortalamasını sabit olarak tahmin eder (Poisson background rate)."""
    return np.full(n_test, float(np.mean(y_train)))


def persistence_forecast(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """y_hat[t] = y[t-1]. İlk test tahmini son eğitim değeridir."""
    prev = np.concatenate([[y_train[-1]], y_test[:-1]])
    return prev.astype(float)
