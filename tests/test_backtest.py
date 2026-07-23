"""ETAS backtest çekirdeği — sızıntısızlık ve pencere aritmetiği."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyequake.forecast.backtest import (
    Backtest,
    InsufficientData,
    backtest_etas,
    etas_forecast,
    to_days,
    window_counts,
    window_starts,
)
from eyequake.forecast.etas import ETASParams, expected_count

PARAMS = ETASParams(mu=0.05, K=0.1, c=0.01, p=1.2, alpha=1.0, m0=4.0, loglik=0.0)


def test_to_days_starts_at_zero_and_scales():
    s = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-03", "2020-02-01"], utc=True))
    d = to_days(s)
    assert d[0] == 0.0
    assert d[1] == pytest.approx(2.0)
    assert d[2] == pytest.approx(31.0)


def test_window_starts_excludes_overflowing_last_window():
    # [0, 100) icinde 30g pencereler: 0, 30, 60 (90+30=120 > 100 disarida)
    assert window_starts(0.0, 100.0, 30.0).tolist() == [0.0, 30.0, 60.0]


def test_window_counts_uses_half_open_interval():
    t = np.array([0.0, 9.9, 10.0, 19.0, 25.0])
    counts = window_counts(t, np.array([0.0, 10.0, 20.0]), 10.0)
    # 10.0 alt sinira dahil, ust sinira degil
    assert counts.tolist() == [2.0, 2.0, 1.0]


def test_etas_forecast_sees_only_past_events():
    t = np.array([0.0, 1.0, 50.0, 51.0])
    m = np.full(4, 5.0)
    starts = np.array([40.0])
    got = etas_forecast(PARAMS, t, m, starts, 10.0)
    # Yalnizca t<40 olan iki olay gecmis sayilir; 50/51 gelecekten sizmamali
    expected = expected_count(PARAMS, t[:2], m[:2], 40.0, 50.0)
    assert got[0] == pytest.approx(expected)


def test_etas_forecast_ignores_future_when_no_history():
    t = np.array([100.0, 101.0])
    m = np.full(2, 5.0)
    got = etas_forecast(PARAMS, t, m, np.array([0.0]), 10.0)
    assert got[0] == pytest.approx(PARAMS.mu * 10.0)  # yalnizca background


def _synthetic_catalog(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 3000.0, size=n))
    m = 4.0 + rng.exponential(0.4, size=n)
    return t, m


def test_backtest_shapes_and_train_test_boundary():
    t, m = _synthetic_catalog()
    bt = backtest_etas(t, m, m0=4.0, window_days=30.0, train_frac=0.75)

    assert isinstance(bt, Backtest)
    assert bt.n_train == int(len(t) * 0.75)
    assert bt.t_split == t[bt.n_train]
    assert bt.starts[0] == bt.t_split  # test train'in gelecegindedir
    n_win = len(bt.starts)
    assert len(bt.actual) == len(bt.etas_pred) == len(bt.climatology) == n_win
    assert len(bt.persistence) == n_win


def test_backtest_climatology_is_flat_and_from_train_only():
    t, m = _synthetic_catalog()
    bt = backtest_etas(t, m, m0=4.0)
    assert len(set(bt.climatology.tolist())) == 1
    train_counts = window_counts(t, window_starts(0.0, bt.t_split, 30.0), 30.0)
    assert bt.climatology[0] == pytest.approx(train_counts.mean())


def test_backtest_persistence_is_lagged_actual():
    t, m = _synthetic_catalog()
    bt = backtest_etas(t, m, m0=4.0)
    assert bt.persistence[1:].tolist() == bt.actual[:-1].tolist()
    assert bt.persistence[0] == pytest.approx(bt.climatology[0])


def test_backtest_raises_on_too_few_events():
    t = np.arange(20.0)
    m = np.full(20, 4.5)
    with pytest.raises(InsufficientData):
        backtest_etas(t, m, m0=4.0)


def test_backtest_raises_when_no_test_window_fits():
    # 200 olay ama toplam sure < pencere -> test penceresi cikmaz
    t = np.linspace(0.0, 5.0, 200)
    m = np.full(200, 4.5)
    with pytest.raises(InsufficientData):
        backtest_etas(t, m, m0=4.0, window_days=30.0)
