"""etas.py karakterizasyon + sınır testleri (causal intensity, integral, branching)."""

from __future__ import annotations

import numpy as np
import pytest

from eyequake.forecast.etas import (
    ETASParams,
    _integral,
    _intensity_at_events,
    branching_ratio,
    evaluate_etas_windows,
    expected_count,
    fit_etas,
)


def test_intensity_first_event_is_background_only():
    t = np.array([0.0, 1.0, 2.0])
    m = np.array([4.0, 4.0, 4.0])
    lam = _intensity_at_events(t, m, mu=0.5, K=0.1, c=0.01, p=1.1, alpha=1.0, m0=4.0)
    assert lam[0] == pytest.approx(0.5)          # ilk olay: yalnızca mu (causal)
    assert lam[1] > 0.5 and lam[2] > 0.5         # önceki olaylar tetikleme ekler


def test_integral_is_background_when_K_zero():
    t = np.array([1.0, 2.0])
    m = np.array([4.0, 4.0])
    val = _integral(t, m, T0=0.0, T1=10.0, mu=0.3, K=0.0, c=0.01, p=1.1, alpha=1.0, m0=4.0)
    assert val == pytest.approx(0.3 * 10.0)


def test_integral_triggering_adds_to_background():
    t = np.array([1.0])
    m = np.array([5.0])
    bg = _integral(t, m, 0.0, 10.0, 0.3, 0.0, 0.01, 1.1, 1.0, 4.0)
    trig = _integral(t, m, 0.0, 10.0, 0.3, 0.2, 0.01, 1.1, 1.0, 4.0)
    assert trig > bg


def test_branching_ratio_none_when_alpha_ge_beta():
    p = ETASParams(mu=0.1, K=0.1, c=0.01, p=1.1, alpha=3.0, m0=4.0, loglik=0.0)
    assert branching_ratio(p, b_value=1.0) is None  # beta=2.30 < 3.0 → tanımsız


def test_branching_ratio_finite_when_alpha_lt_beta():
    p = ETASParams(mu=0.1, K=0.05, c=0.02, p=1.2, alpha=1.0, m0=4.0, loglik=0.0)
    n = branching_ratio(p, b_value=1.0)
    assert n is not None and n > 0


def test_expected_count_zero_history_is_background():
    p = ETASParams(mu=0.2, K=0.1, c=0.01, p=1.1, alpha=1.0, m0=4.0, loglik=0.0)
    n = expected_count(p, np.array([]), np.array([]), 0.0, 10.0)
    assert n == pytest.approx(0.2 * 10.0)


def test_fit_etas_returns_valid_params():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 365, size=150))
    m = 4.0 + rng.exponential(0.4, 150)
    params = fit_etas(t, m, m0=4.0, T0=0.0, T1=365.0)
    assert params.mu > 0 and params.K >= 0
    assert 1.01 <= params.p <= 3.0       # bounds içinde
    assert np.isfinite(params.loglik)


def _poisson_catalog(n: int = 300, days: float = 2000.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, days, size=n))
    m = 4.0 + rng.exponential(1.0 / np.log(10.0), size=n)  # GR, b=1
    return t, m


def test_evaluate_etas_windows_is_out_of_time_and_shapes_align():
    t, m = _poisson_catalog()
    ev = evaluate_etas_windows(t, m, m0=4.0, train_frac=0.75, window_days=30.0)
    assert ev.n_train_events == int(len(t) * 0.75)
    assert ev.t_split == pytest.approx(t[ev.n_train_events])
    n = ev.actual.size
    assert n > 0
    assert ev.etas_pred.shape == ev.climatology.shape == ev.persistence.shape == (n,)
    # Test pencereleri yalnızca split'ten SONRAKİ olayları sayar.
    assert ev.actual.sum() <= np.sum(t >= ev.t_split)
    assert np.isfinite(ev.skill_vs_climatology)
    assert np.isfinite(ev.skill_vs_persistence)


def test_evaluate_etas_windows_does_not_fake_skill_on_poisson_data():
    """Saf Poisson katalogda ETAS climatology'yi anlamlı biçimde yenmemeli.

    Kümelenme yoksa tetikleme terimi bilgi taşımaz; büyük pozitif beceri burada
    sızıntı (leakage) ya da hatalı baseline işareti olurdu.
    """
    t, m = _poisson_catalog(n=400, days=3000.0, seed=3)
    ev = evaluate_etas_windows(t, m, m0=4.0)
    assert ev.skill_vs_climatology < 0.2


def test_evaluate_etas_windows_climatology_is_train_window_mean():
    t, m = _poisson_catalog()
    ev = evaluate_etas_windows(t, m, m0=4.0, window_days=30.0)
    starts = np.arange(0.0, ev.t_split - 30.0, 30.0)
    expected = np.mean([np.sum((t >= s) & (t < s + 30.0)) for s in starts])
    assert np.allclose(ev.climatology, expected)


@pytest.mark.parametrize(
    "t, m, kwargs",
    [
        (np.array([0.0]), np.array([4.0]), {}),                       # tek olay
        (np.array([0.0, 1.0, 2.0]), np.array([4.0, 4.0]), {}),        # uzunluk uyuşmuyor
        (np.array([2.0, 1.0, 3.0]), np.array([4.0, 4.0, 4.0]), {}),   # sırasız
        (np.linspace(0, 10, 20), np.full(20, 4.0), {"window_days": 30.0}),  # pencere yok
        (np.linspace(0, 900, 50), np.full(50, 4.0), {"train_frac": 0.0}),
    ],
)
def test_evaluate_etas_windows_rejects_unmeasurable_setups(t, m, kwargs):
    with pytest.raises(ValueError):
        evaluate_etas_windows(t, m, m0=4.0, **kwargs)
