"""forecast/spatial_skill.py testleri — uzaysal öngörü-beceri metriği.

Tanımlayıcı beceri ölçüsü (deprem tahmini değil): eğitim penceresinden kurulan
risk yüzeyi, gelecekteki (test) olayları alan-uniform baseline'a göre daha iyi
yoğunlaştırıyor mu?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyequake.config import TURKEY
from eyequake.forecast.spatial_skill import (
    area_skill_score,
    evaluate_surface_skill,
    spatial_concentration_gain,
)


def test_perfect_concentration_scores_near_one():
    # Gerçek maksimum yoğunlaşma: tüm olaylar tek (en yüksek riskli) hücrede.
    # Sonlu-n için metrik tavanı (n-1)/n = 7/8 = 0.875 ("bire yakın").
    counts = np.array([23, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    assert area_skill_score(counts) > 0.7


def test_uniform_concentration_scores_near_zero():
    counts = np.array([3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
    assert abs(area_skill_score(counts)) < 1e-9


def test_anti_concentration_scores_negative():
    # Tam ters yoğunlaşma: tüm olaylar en düşük riskli hücrede → -(n-1)/n = -0.875.
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 23], dtype=float)
    assert area_skill_score(counts) < -0.7


def test_no_events_raises():
    with pytest.raises(ValueError):
        area_skill_score(np.zeros(5))


def test_gain_top_decile_beats_uniform():
    counts = np.array([10, 6, 2, 2, 0, 0, 0, 0], dtype=float)
    assert spatial_concentration_gain(counts, area_fraction=0.25) > 1.0


def _two_cluster_catalog(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """İki sıcak-nokta etrafında sentetik küme katalogu (out-of-time test için)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[39.0, 28.0], [37.0, 36.0]])
    which = rng.integers(0, 2, size=n)
    lat = centers[which, 0] + rng.normal(0.0, 0.1, size=n)
    lon = centers[which, 1] + rng.normal(0.0, 0.1, size=n)
    mag = rng.uniform(4.0, 6.0, size=n)
    base = pd.Timestamp("2000-01-01", tz="UTC")
    span_days = (pd.Timestamp("2020-01-01", tz="UTC") - base).days
    offsets = rng.uniform(0.0, float(span_days), size=n)
    time = base + pd.to_timedelta(offsets, unit="D")
    return pd.DataFrame({"time": time, "latitude": lat, "longitude": lon, "mag": mag})


def test_out_of_time_skill_on_clustered_synthetic_catalog():
    df = _two_cluster_catalog()
    result = evaluate_surface_skill(df, TURKEY, min_mag=4.0, test_frac=0.25)
    assert result["n_test_events"] > 0
    assert result["area_skill_score"] > 0.2
