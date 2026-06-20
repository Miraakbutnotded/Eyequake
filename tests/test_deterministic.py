"""deterministic.py karakterizasyon testleri (olay-özelliği, leakage-yok)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eyequake.forecast.deterministic import build_event_features


def _catalog(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "time": pd.date_range("2000-01-01", periods=n, freq="10D", tz="UTC"),
        "mag": 4.0 + rng.random(n),
        "latitude": 39.0,
        "longitude": 35.0,
    })


def test_feature_shape_and_names():
    k = 10
    f = build_event_features(_catalog(30), min_mag=4.0, k=k)
    assert f.X.shape == (30 - k, 6)
    assert f.y.shape == (30 - k,)
    assert len(f.feature_names) == 6
    assert f.feature_names[0] == "last_mag"


def test_target_is_current_and_features_are_past_only():
    df = _catalog(20)
    k = 5
    f = build_event_features(df, min_mag=4.0, k=k)
    mags = df.sort_values("time")["mag"].to_numpy()
    assert f.y[0] == mags[k]          # hedef = o olayın büyüklüğü
    assert f.X[0, 0] == mags[k - 1]   # last_mag = bir önceki olay (leakage yok)


def test_deterministic_repeatable():
    df = _catalog(25)
    f1 = build_event_features(df, k=8)
    f2 = build_event_features(df, k=8)
    assert np.allclose(f1.X, f2.X) and np.allclose(f1.y, f2.y)
