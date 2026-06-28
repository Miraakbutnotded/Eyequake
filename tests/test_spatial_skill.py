"""forecast/spatial_skill.py testleri — uzaysal öngörü-beceri metriği.

Tanımlayıcı beceri ölçüsü (deprem tahmini değil): eğitim penceresinden kurulan
risk yüzeyi, gelecekteki (test) olayları alan-uniform baseline'a göre daha iyi
yoğunlaştırıyor mu?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyequake.config import TURKEY
from eyequake.forecast.spatial_skill import (
    area_skill_score,
    evaluate_surface_skill,
    spatial_concentration_gain,
)

# Served-surface params (scripts/06_export_web_data.py): the skill claim shipped to
# the product is measured at THESE values, not the build_cell_index defaults.
SERVED_MIN_MAG = 3.0
SERVED_CELL_DEG = 0.25
CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "turkey_catalog.csv"

# --- area_skill_score: yoğunlaşma uçları ---


def test_perfect_concentration_scores_near_one():
    # Gerçek maksimum yoğunlaşma: tüm olaylar tek (en yüksek riskli) hücrede.
    # Sonlu-n için metrik tavanı (n-1)/n = 7/8 = 0.875 ("bire yakın").
    counts = np.array([23, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    assert area_skill_score(counts) > 0.7
    assert area_skill_score(counts) == pytest.approx((8 - 1) / 8)  # 0.875 tavan


def test_uniform_concentration_scores_near_zero():
    counts = np.array([3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
    assert abs(area_skill_score(counts)) < 1e-9


def test_anti_concentration_scores_negative():
    # Tam ters yoğunlaşma: tüm olaylar en düşük riskli hücrede → -(n-1)/n = -0.875.
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 23], dtype=float)
    assert area_skill_score(counts) < -0.7
    assert area_skill_score(counts) == pytest.approx(-(8 - 1) / 8)  # -0.875 taban


def test_single_cell_scores_zero():
    # Tek hücre: trapez y=x doğrusu → ASS = 0 (yoğunlaşma tanımsız/nötr).
    assert area_skill_score(np.array([5.0])) == 0.0


def test_no_events_raises():
    with pytest.raises(ValueError):
        area_skill_score(np.zeros(5))


def test_nan_input_raises_area_skill_score():
    with pytest.raises(ValueError):
        area_skill_score(np.array([1.0, np.nan, 2.0]))


def test_negative_input_raises_area_skill_score():
    with pytest.raises(ValueError):
        area_skill_score(np.array([3.0, -1.0, 2.0]))


# --- spatial_concentration_gain ---


def test_gain_top_quartile_beats_uniform():
    counts = np.array([10, 6, 2, 2, 0, 0, 0, 0], dtype=float)
    assert spatial_concentration_gain(counts, area_fraction=0.25) > 1.0


def test_gain_is_one_on_uniform_field():
    # Sıfır-beceri uniform alan TAM 1.0 vermeli — taban k/n'ye sabitlenmeli,
    # istenen area_fraction'a değil (k/n yuvarlama nedeniyle 0.25'ten sapabilir).
    for n in (5, 6, 10, 11, 14):
        counts = np.full(n, 3.0)
        assert spatial_concentration_gain(counts, 0.25) == pytest.approx(1.0)


def test_gain_invalid_area_fraction_raises():
    counts = np.array([5.0, 1.0, 0.0, 0.0], dtype=float)
    with pytest.raises(ValueError):
        spatial_concentration_gain(counts, area_fraction=0.0)
    with pytest.raises(ValueError):
        spatial_concentration_gain(counts, area_fraction=1.5)


def test_gain_no_events_raises():
    with pytest.raises(ValueError):
        spatial_concentration_gain(np.zeros(5), area_fraction=0.25)


def test_nan_input_raises_gain():
    with pytest.raises(ValueError):
        spatial_concentration_gain(np.array([1.0, np.nan]), area_fraction=0.25)


def test_negative_input_raises_gain():
    with pytest.raises(ValueError):
        spatial_concentration_gain(np.array([-1.0, 2.0]), area_fraction=0.25)


# --- evaluate_surface_skill: out-of-time entegrasyon + koruma dalları ---


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


def test_empty_catalog_raises():
    empty = pd.DataFrame({"time": [], "latitude": [], "longitude": [], "mag": []})
    with pytest.raises(ValueError):
        evaluate_surface_skill(empty, TURKEY)


def test_empty_test_window_raises():
    # Tüm zamanlar aynı → kuantil bölme test penceresini boş bırakır.
    same = pd.DataFrame(
        {
            "time": pd.to_datetime(["2010-01-01"] * 6, utc=True),
            "latitude": [39.0] * 6,
            "longitude": [28.0] * 6,
            "mag": [5.0] * 6,
        }
    )
    with pytest.raises(ValueError):
        evaluate_surface_skill(same, TURKEY)


def test_empty_train_window_raises():
    # Tüm zamanlar NaT → kuantil NaT → times<=NaT tümü False → eğitim boş.
    nat = pd.DataFrame(
        {
            "time": pd.Series([pd.NaT] * 6, dtype="datetime64[ns]"),
            "latitude": [39.0] * 6,
            "longitude": [28.0] * 6,
            "mag": [5.0] * 6,
        }
    )
    with pytest.raises(ValueError):
        evaluate_surface_skill(nat, TURKEY)


def test_test_events_outside_train_cells_returns_none():
    # Eğitim olayları erken+A kümesinde; test olayları geç+uzak B kümesinde →
    # hiçbir test olayı eğitim hücrelerine düşmez → skill/gain None.
    times = [
        "2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01", "2005-01-01",
        "2006-01-01", "2007-01-01", "2008-01-01", "2009-01-01",
        "2018-01-01", "2019-01-01", "2020-01-01",
    ]
    lat = [39.0] * 9 + [41.0] * 3
    lon = [28.0] * 9 + [44.0] * 3
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "latitude": lat,
            "longitude": lon,
            "mag": [5.0] * 12,
        }
    )
    result = evaluate_surface_skill(df, TURKEY, min_mag=4.0, test_frac=0.25)
    assert result["n_test_events"] == 0
    assert result["area_skill_score"] is None
    assert result["gain_top25"] is None


# --- Real served-surface regression: the SHIPPED surface beats area-uniform Poisson ---


def test_real_catalog_served_surface_beats_baseline():
    """The real AFAD surface, at the SERVED params (M≥3.0, 0.25°), must have skill > 0.

    This is the empirical anchor behind the stamped meta.area_skill_score and the
    validate_science gate. The catalog is .gitignored (data/processed/) — skip when
    absent (e.g. CI), where the gate logic is covered by synthetic-catalog tests and
    the committed-surface contract. CRITICAL: measured at min_mag=3.0 (scripts/06
    MIN_MAG), not the build_cell_index default — a wrong threshold measures a
    different surface.
    """
    if not CATALOG_PATH.exists():
        pytest.skip(f"Katalog yok (gitignored): {CATALOG_PATH}")

    df = pd.read_csv(CATALOG_PATH, parse_dates=["time"])
    result = evaluate_surface_skill(
        df, TURKEY, min_mag=SERVED_MIN_MAG, cell_deg=SERVED_CELL_DEG
    )

    assert result["n_test_events"] > 0
    assert result["area_skill_score"] is not None
    # Surface genuinely concentrates future events better than a uniform baseline.
    assert result["area_skill_score"] > 0
    assert result["gain_top25"] > 1.0
