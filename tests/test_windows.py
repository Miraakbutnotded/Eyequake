"""windows.py sözleşme testleri: pencereleme, lag kurulumu ve zaman sırası koruması.

Kritik nokta: make_supervised gelecekten geçmişe veri sızdırmamalı (leakage). Sızıntı,
sahte yüksek beceri üretir ve baseline karşılaştırmasını geçersiz kılar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyequake.forecast.windows import Supervised, build_windowed_series, make_supervised


def _catalog(times: list[str], mags: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"time": pd.to_datetime(times, utc=True), "mag": mags})


def test_windowing_counts_events_and_tracks_max_magnitude():
    df = _catalog(
        ["2020-01-02", "2020-01-10", "2020-02-05"],
        [4.5, 5.2, 4.1],
    )
    out = build_windowed_series(df, freq_days=30, min_mag=4.0)
    assert out["n_events"].tolist() == [2, 1]
    assert out["max_mag"].tolist() == [5.2, 4.1]


def test_windowing_filters_below_min_magnitude():
    df = _catalog(["2020-01-02", "2020-01-03"], [3.9, 4.0])
    out = build_windowed_series(df, freq_days=30, min_mag=4.0)
    assert out["n_events"].sum() == 1  # 3.9 eşiğin altında, sayılmaz


def test_windowing_marks_quiet_windows_as_zero_not_nan():
    # Ortadaki pencerede olay yok: n_events=0, max_mag=0.0 olmalı (NaN değil).
    df = _catalog(["2020-01-01", "2020-04-01"], [5.0, 5.0])
    out = build_windowed_series(df, freq_days=30, min_mag=4.0)
    assert not out["max_mag"].isna().any()
    assert (out["n_events"] == 0).any()
    assert out["n_events"].dtype == np.int64  # sayım tam sayı kalmalı


def test_windowing_parses_string_timestamps():
    df = pd.DataFrame({"time": ["2020-01-02T00:00:00Z", "2020-01-09T00:00:00Z"], "mag": [4.5, 4.6]})
    out = build_windowed_series(df, freq_days=30, min_mag=4.0)
    assert out["n_events"].tolist() == [2]


def test_windowing_empty_history_returns_empty_frame_with_schema():
    """Eşiği geçen hiç olay yoksa boş ama şema-uyumlu çerçeve dönmeli."""
    df = _catalog(["2020-01-02"], [3.0])
    out = build_windowed_series(df, freq_days=30, min_mag=4.0)
    assert len(out) == 0
    assert list(out.columns) == ["n_events", "max_mag"]


def _series(n_events: list[int], max_mag: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(n_events), freq="30D", tz="UTC")
    return pd.DataFrame({"n_events": n_events, "max_mag": max_mag}, index=idx)


def test_supervised_shapes_and_feature_names_align():
    series = _series([1, 2, 3, 4, 5], [4.0, 4.1, 4.2, 4.3, 4.4])
    sup = make_supervised(series, n_lags=2)
    assert isinstance(sup, Supervised)
    assert sup.X.shape == (3, 4)  # 5 pencere - 2 lag = 3 satır; 2*n_lags sütun
    assert sup.y.shape == (3,)
    assert sup.feature_names == [
        "n_events_lag1",
        "n_events_lag2",
        "max_mag_lag1",
        "max_mag_lag2",
    ]
    assert sup.X.shape[1] == len(sup.feature_names)


def test_supervised_lag1_is_the_immediately_previous_window():
    series = _series([10, 20, 30, 40], [4.0, 4.1, 4.2, 4.3])
    sup = make_supervised(series, n_lags=2)
    # İlk satır t=2: lag1 = pencere 1 (20), lag2 = pencere 0 (10).
    assert sup.X[0, 0] == 20.0
    assert sup.X[0, 1] == 10.0
    assert sup.y[0] == 30.0


def test_supervised_never_leaks_the_target_window():
    """Sızıntı koruması: X satırı t yalnızca t'den ÖNCEKİ pencereleri içerebilir."""
    series = _series([1, 2, 3, 4, 5, 6], [4.0, 4.1, 4.2, 4.3, 4.4, 4.5])
    n_lags = 3
    sup = make_supervised(series, n_lags=n_lags)
    counts = series["n_events"].to_numpy(dtype=float)
    for row, target_idx in enumerate(range(n_lags, len(series))):
        lags = sup.X[row, :n_lags]
        expected = counts[target_idx - n_lags : target_idx][::-1]
        assert lags.tolist() == expected.tolist()
        assert counts[target_idx] not in lags.tolist() or counts[target_idx] in expected.tolist()
        assert sup.window_starts[row] == series.index[target_idx]


def test_supervised_window_starts_track_the_target_not_the_lags():
    series = _series([1, 2, 3, 4], [4.0, 4.1, 4.2, 4.3])
    sup = make_supervised(series, n_lags=2)
    assert list(sup.window_starts) == list(series.index[2:])


def test_supervised_supports_max_mag_target():
    series = _series([1, 2, 3, 4], [4.0, 4.1, 4.2, 4.9])
    sup = make_supervised(series, n_lags=2, target="max_mag")
    assert sup.y.tolist() == [4.2, 4.9]


@pytest.mark.parametrize("n_windows", [0, 1, 5])
def test_supervised_keeps_2d_shape_when_history_is_too_short(n_windows):
    """Yetersiz geçmişte X boş AMA 2-B kalmalı.

    np.asarray([]) (0,) şeklinde 1-B dizi verirdi; X.shape[1] veya model.fit
    bekleyen downstream kod anlaşılmaz bir hatayla düşerdi.
    """
    series = _series([1] * n_windows, [4.0] * n_windows)
    sup = make_supervised(series, n_lags=6)
    assert sup.X.ndim == 2
    assert sup.X.shape == (0, 12)
    assert sup.y.shape == (0,)
    assert len(sup.window_starts) == 0
