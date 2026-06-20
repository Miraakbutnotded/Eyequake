"""evaluate.py karakterizasyon + sınır testleri (leakage-yok, Poisson deviance)."""

from __future__ import annotations

import numpy as np
import pytest

from eyequake.forecast.evaluate import (
    bootstrap_mean_ci,
    poisson_deviance,
    score,
    skill_score,
    temporal_split,
)


def test_temporal_split_sizes_and_no_leakage():
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20.0)
    s = temporal_split(X, y, test_frac=0.25)
    assert len(s.y_train) == 15 and len(s.y_test) == 5
    assert s.X_train[-1, 0] < s.X_test[0, 0]  # test her zaman eğitimin geleceğinde


def test_poisson_deviance_perfect_is_zero():
    y = np.array([0.0, 1, 2, 3, 10])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-8)


def test_poisson_deviance_zero_obs_safe():
    d = poisson_deviance(np.array([0.0, 0, 0]), np.array([1.0, 2, 3]))
    assert np.isfinite(d) and d > 0


def test_score_keys_and_mae():
    s = score(np.array([1.0, 2, 3]), np.array([2.0, 2, 2]))
    assert set(s) == {"mae", "rmse", "poisson_deviance"}
    assert s["mae"] == pytest.approx((1 + 0 + 1) / 3)


def test_skill_score_baseline_zero_returns_zero():
    assert skill_score(0.5, 0.0) == 0.0


def test_skill_score_improvement_and_regression():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)   # %50 iyileşme
    assert skill_score(1.5, 1.0) == pytest.approx(-0.5)  # baseline'dan kötü


def test_bootstrap_mean_ci_brackets_mean_and_deterministic():
    x = np.arange(100.0)
    lo, mean, hi = bootstrap_mean_ci(x, n_boot=1000, seed=0)
    assert lo < mean < hi
    assert mean == pytest.approx(49.5)
    assert bootstrap_mean_ci(x, seed=0) == bootstrap_mean_ci(x, seed=0)  # tekrarlanabilir
