"""Temel sismoloji istatistikleri: FMD, magnitude of completeness, b-değeri.

Bu fonksiyonlar bir kataloğun analiz için **kullanılabilir** olup olmadığını
belirler. Tamamlanmamış bir katalogda (Mc'nin altında) yapılan her tahmin,
sismisiteyi değil raporlama yanlılığını modeller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LOG10_E = math.log10(math.e)  # 0.4342944819...


@dataclass(frozen=True)
class FMD:
    """Frequency-Magnitude Distribution (Gutenberg-Richter)."""

    bin_centers: np.ndarray
    incremental: np.ndarray  # her büyüklük binindeki olay sayısı
    cumulative: np.ndarray  # M >= bin olay sayısı


@dataclass(frozen=True)
class BValue:
    """Aki maksimum olabilirlik b-değeri + Shi & Bolt belirsizliği."""

    b: float
    sigma: float
    a: float  # GR a-değeri (log10 toplam oran), Mc'de normalize
    mc: float
    n_above_mc: int


def frequency_magnitude_distribution(mags: np.ndarray, bin_width: float = 0.1) -> FMD:
    """Büyüklük binlerine göre artımlı ve kümülatif olay sayıları."""
    mags = np.asarray(mags, dtype=float)
    lo = math.floor(mags.min() / bin_width) * bin_width
    # Kenarları tam-sayı bin sayısından kur (np.arange float kayması en büyük depremi
    # FMD'den düşürebiliyordu). Üst kenar maks büyüklüğün kesin üstünde olur.
    n_bins = int(round((mags.max() - lo) / bin_width)) + 1
    edges = lo + bin_width * np.arange(n_bins + 1)
    incremental, _ = np.histogram(mags, bins=edges)
    centers = edges[:-1] + bin_width / 2.0
    # Kümülatif: M >= center (yüksekten alçağa toplam).
    cumulative = np.cumsum(incremental[::-1])[::-1]
    return FMD(bin_centers=centers, incremental=incremental, cumulative=cumulative)


def magnitude_of_completeness(mags: np.ndarray, bin_width: float = 0.1) -> float:
    """Maksimum eğrilik (maximum curvature) yöntemiyle Mc.

    FMD'nin en çok olay içeren büyüklük binini bulur; literatürde yaygın olan
    +0.2 düzeltmesini uygular (Woessner & Wiemer 2005).
    """
    fmd = frequency_magnitude_distribution(mags, bin_width)
    maxc = float(fmd.bin_centers[int(np.argmax(fmd.incremental))])
    return round(maxc + 0.2, 2)


def b_value_aki(mags: np.ndarray, mc: float, bin_width: float = 0.1) -> BValue:
    """Aki (1965) MLE b-değeri, Mc üstündeki olaylarla.

    b = log10(e) / (mean(M) - (Mc - dM/2))
    Belirsizlik: Shi & Bolt (1982).
    """
    mags = np.asarray(mags, dtype=float)
    above = mags[mags >= mc - 1e-9]
    n = int(above.size)
    if n < 2:
        raise ValueError(f"Mc={mc} üstünde yeterli olay yok (n={n}).")

    mbar = float(above.mean())
    denom = mbar - (mc - bin_width / 2.0)
    if denom <= 0:
        raise ValueError("Geçersiz b-değeri paydası; Mc fazla yüksek olabilir.")

    b = _LOG10_E / denom
    sigma = 2.30 * b**2 * math.sqrt(float(np.sum((above - mbar) ** 2)) / (n * (n - 1)))
    a = math.log10(n) + b * mc
    return BValue(b=round(b, 3), sigma=round(sigma, 3), a=round(a, 3), mc=mc, n_above_mc=n)
