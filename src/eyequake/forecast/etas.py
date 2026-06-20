"""Track B (düzgün) — Temporal ETAS modeli.

ETAS (Epidemic-Type Aftershock Sequence): sismisite oranının fiziksel-temelli
olasılıksal modeli. Koşullu yoğunluk (conditional intensity):

    λ(t) = μ + Σ_{t_i < t}  K · e^{α(m_i − m0)} · (t − t_i + c)^(−p)

- μ        : arka plan (background) oranı
- K, α     : tetikleme verimliliği (büyük deprem → daha çok artçı)
- c, p     : Omori-Utsu zamansal sönüm
- m0       : referans (tamlık) büyüklüğü

Parametreler, olay zamanları + büyüklükleri üzerinde maksimum olabilirlikle (MLE)
kestirilir. Bu, "şu dönemde beklenen deprem oranı" için savunulabilir olasılıksal
çıktıdır — deterministik tahmin DEĞİL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ETASParams:
    mu: float       # background oranı (olay/gün)
    K: float        # tetikleme verimliliği
    c: float        # Omori zaman kayması (gün)
    p: float        # Omori sönüm üssü
    alpha: float    # büyüklük-verimlilik katsayısı
    m0: float       # referans büyüklük
    loglik: float   # fit log-olabilirliği


def _intensity_at_events(t: np.ndarray, m: np.ndarray, mu, K, c, p, alpha, m0) -> np.ndarray:
    """Her olay zamanındaki λ (yalnızca önceki olaylardan tetikleme).

    Karmaşıklık O(n²). Bilinçli olarak M≥4.0 alt-kataloğunda (~4.2k olay) kullanılır;
    orada saniyeler içinde koşar. Tüm AFAD kataloğu (156k, M≥2.5) için tasarlanmamıştır
    — gerekirse zaman-pencereli tetikleme kesmesi (yakın geçmişle sınırla) eklenmelidir.
    """
    lam = np.full(t.size, mu, dtype=float)
    for i in range(1, t.size):
        dt = t[i] - t[:i]
        lam[i] += np.sum(K * np.exp(alpha * (m[:i] - m0)) * (dt + c) ** (-p))
    return lam


def _integral(t: np.ndarray, m: np.ndarray, T0, T1, mu, K, c, p, alpha, m0) -> float:
    """∫_{T0}^{T1} λ(t) dt — beklenen toplam olay sayısı."""
    bg = mu * (T1 - T0)
    # Her olayın [t_i, T1] üzerindeki Omori katkısı (p≠1).
    upper = (T1 - t + c) ** (1.0 - p)
    lower = c ** (1.0 - p)
    trig = np.sum(K * np.exp(alpha * (m - m0)) * (upper - lower) / (1.0 - p))
    return bg + trig


def _neg_loglik(params, t, m, T0, T1, m0) -> float:
    mu, K, c, p, alpha = params
    lam = _intensity_at_events(t, m, mu, K, c, p, alpha, m0)
    if np.any(lam <= 0):
        return 1e12
    ll = np.sum(np.log(lam)) - _integral(t, m, T0, T1, mu, K, c, p, alpha, m0)
    return -ll if np.isfinite(ll) else 1e12


def fit_etas(t: np.ndarray, m: np.ndarray, m0: float, T0: float, T1: float) -> ETASParams:
    """Temporal ETAS parametrelerini MLE ile kestirir (L-BFGS-B + sınırlar)."""
    span = max(T1 - T0, 1.0)
    x0 = [t.size / span * 0.5, 0.1, 0.01, 1.1, 1.0]  # mu, K, c, p, alpha
    bounds = [(1e-6, None), (1e-6, None), (1e-4, 5.0), (1.01, 3.0), (0.1, 4.0)]
    res = minimize(
        _neg_loglik, x0, args=(t, m, T0, T1, m0),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )
    mu, K, c, p, alpha = res.x
    return ETASParams(mu=mu, K=K, c=c, p=p, alpha=alpha, m0=m0, loglik=-res.fun)


def expected_count(
    params: ETASParams, hist_t: np.ndarray, hist_m: np.ndarray,
    window_start: float, window_end: float,
) -> float:
    """[window_start, window_end] aralığında beklenen olay sayısı.

    hist_t/hist_m: pencere başından ÖNCE gerçekleşmiş olaylar (forecast bilgisi).
    """
    p, c, K, alpha, m0 = params.p, params.c, params.K, params.alpha, params.m0
    bg = params.mu * (window_end - window_start)
    if hist_t.size == 0:
        return bg
    upper = (window_end - hist_t + c) ** (1.0 - p)
    lower = (window_start - hist_t + c) ** (1.0 - p)
    trig = np.sum(K * np.exp(alpha * (hist_m - m0)) * (upper - lower) / (1.0 - p))
    return float(bg + trig)


def branching_ratio(params: ETASParams, b_value: float) -> float | None:
    """Dallanma oranı n (olay başına beklenen tetiklenmiş olay). α<b·ln10 ise tanımlı.

    n = K·c^(1−p)/(p−1) · (β/(β−α)),  β = b·ln10
    n≥1 kritik/süper-kritik (sönmeyen kaskad) anlamına gelir.
    """
    beta = b_value * np.log(10.0)
    if params.alpha >= beta or params.p <= 1:
        return None
    omori = params.c ** (1.0 - params.p) / (params.p - 1.0)
    return float(params.K * omori * beta / (beta - params.alpha))
