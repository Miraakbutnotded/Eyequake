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

from .baselines import climatology_forecast, persistence_forecast
from .evaluate import score, skill_score


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


@dataclass(frozen=True)
class ETASEvaluation:
    """Zaman-dışı ETAS değerlendirmesi: test pencerelerinde beklenen vs gerçek sayım."""

    params: ETASParams
    t_split: float              # train/test sınırı (gün)
    n_train_events: int
    actual: np.ndarray          # test penceresi başına gerçek olay sayısı
    etas_pred: np.ndarray       # yalnızca pencere-öncesi geçmişten ETAS beklentisi
    climatology: np.ndarray     # train pencere ortalaması (Poisson background)
    persistence: np.ndarray     # önceki pencerenin gerçeği
    skill_vs_climatology: float  # Poisson deviance beceri skoru (>0 → yendi)
    skill_vs_persistence: float


def evaluate_etas_windows(
    t: np.ndarray, m: np.ndarray, *, m0: float,
    train_frac: float = 0.75, window_days: float = 30.0,
) -> ETASEvaluation:
    """ETAS'ı train'de fit eder, test pencerelerinde baseline'lara karşı ölçer.

    Tek doğruluk kaynağı: `scripts/08_etas.py` (rapor) ve `scripts/11_export_etas_
    params.py` (web'e stamplanan beceri) AYNI hesabı kullanır — servis edilen beceri
    sayısı elle yazılmış bir literal değil, her koşuda yeniden ölçülen değerdir.

    t: olay zamanları (gün, artan sıralı); m: büyüklükler.
    Yetersiz train/test penceresinde ValueError (sessiz NaN beceri yerine).
    """
    t = np.asarray(t, dtype=float)
    m = np.asarray(m, dtype=float)
    if t.size != m.size:
        raise ValueError("t ve m aynı uzunlukta olmalı")
    if t.size < 2 or np.any(np.diff(t) < 0):
        raise ValueError("t en az 2 olay içermeli ve artan sıralı olmalı")
    k = int(t.size * train_frac)
    if not 0 < k < t.size:
        raise ValueError(f"train_frac={train_frac} geçerli bir train/test bölünmesi vermiyor")
    t_split, t_end = float(t[k]), float(t[-1])

    win_starts = np.arange(t_split, t_end - window_days, window_days)
    train_starts = np.arange(0.0, t_split - window_days, window_days)
    if win_starts.size == 0 or train_starts.size == 0:
        raise ValueError(
            f"{window_days:g} günlük pencerelerle train ({train_starts.size}) veya test "
            f"({win_starts.size}) penceresi kurulamadı; beceri ölçülemez."
        )

    params = fit_etas(t[:k], m[:k], m0=m0, T0=0.0, T1=t_split)

    actual, etas_pred = [], []
    for ws in win_starts:
        we = ws + window_days
        actual.append(int(np.sum((t >= ws) & (t < we))))
        hist = t < ws
        etas_pred.append(expected_count(params, t[hist], m[hist], ws, we))
    actual_arr = np.asarray(actual, dtype=float)
    etas_arr = np.asarray(etas_pred, dtype=float)

    train_counts = np.asarray(
        [np.sum((t >= s) & (t < s + window_days)) for s in train_starts], dtype=float
    )
    clim = climatology_forecast(train_counts, actual_arr.size)
    pers = persistence_forecast(train_counts, actual_arr)

    dev = {
        name: score(actual_arr, pred)["poisson_deviance"]
        for name, pred in (("etas", etas_arr), ("clim", clim), ("pers", pers))
    }
    return ETASEvaluation(
        params=params,
        t_split=t_split,
        n_train_events=k,
        actual=actual_arr,
        etas_pred=etas_arr,
        climatology=clim,
        persistence=pers,
        skill_vs_climatology=skill_score(dev["etas"], dev["clim"]),
        skill_vs_persistence=skill_score(dev["etas"], dev["pers"]),
    )


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
