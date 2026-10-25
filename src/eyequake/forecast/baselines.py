"""Dürüst baseline'lar. Bir modelin anlamlı olması için bunları YENMESİ gerekir.

- Climatology / Poisson: eğitim ortalamasını sabit tahmin et. Deprem oranının
  hafızasız (memoryless) olduğu varsayımının karşılığı. Yenilmesi en zor olan.
- Persistence: bir sonraki pencere = önceki pencere. Otokorelasyon varsa güçlü.
"""

from __future__ import annotations

import numpy as np


def _require_history(y_train: np.ndarray) -> np.ndarray:
    """Boş eğitim geçmişini SESSİZ NaN yerine açık hatayla reddeder.

    `np.mean([])` NaN döndürür; NaN bir baseline, skill_score üzerinden NaN beceriye
    yayılır ve IEEE-754 gereği her `<=` karşılaştırması False olduğu için dürüstlük
    kapılarından "baseline yenildi" gibi geçer. Baseline, her modelin ölçüldüğü
    referans olduğundan burada fail-closed davranmak zorunlu.
    """
    arr = np.asarray(y_train, dtype=float)
    if arr.size == 0:
        raise ValueError(
            "Boş eğitim geçmişi: baseline kurulamaz. Baseline tanımsızsa hiçbir model "
            "'baseline'ı yendi' iddiasında bulunamaz."
        )
    return arr


def climatology_forecast(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Eğitim ortalamasını sabit olarak tahmin eder (Poisson background rate)."""
    return np.full(n_test, float(np.mean(_require_history(y_train))))


def persistence_forecast(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """y_hat[t] = y[t-1]. İlk test tahmini son eğitim değeridir."""
    train = _require_history(y_train)
    prev = np.concatenate([[train[-1]], np.asarray(y_test, dtype=float)[:-1]])
    return prev.astype(float)
