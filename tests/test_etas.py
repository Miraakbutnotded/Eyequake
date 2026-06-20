"""etas.py karakterizasyon + sınır testleri (causal intensity, integral, branching)."""

from __future__ import annotations

import numpy as np
import pytest

from eyequake.forecast.etas import (
    ETASParams,
    _integral,
    _intensity_at_events,
    branching_ratio,
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
