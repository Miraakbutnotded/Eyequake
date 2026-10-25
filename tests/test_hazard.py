"""hazard.py karakterizasyon testleri (uzaysal grid sayım + oran)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eyequake.analysis.hazard import spatial_grid_rates
from eyequake.config import Region

REGION = Region("test", min_lat=38.0, max_lat=42.0, min_lon=30.0, max_lon=36.0)


def _catalog() -> pd.DataFrame:
    # 3 olay aynı hücrede, 1 olay başka hücrede, 1 olay bbox DIŞI (50°N), 1 olay M<4.
    return pd.DataFrame({
        "time": pd.to_datetime(
            ["2000-01-01", "2005-01-01", "2010-01-01", "2010-01-01", "2010-01-01", "2008-01-01"],
            utc=True,
        ),
        "latitude": [39.1, 39.1, 39.1, 40.6, 50.0, 39.1],
        "longitude": [35.1, 35.1, 35.1, 30.1, 35.0, 35.1],
        "mag": [4.5, 4.6, 5.0, 4.2, 6.0, 3.0],
    })


def test_counts_sum_only_in_bbox_above_min_mag():
    g = spatial_grid_rates(_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    # bbox içi M≥4 = 4 olay (50°N bbox dışı, M3.0 eşik altı)
    assert int(g.counts.sum()) == 4


def test_top_cell_has_three_events():
    g = spatial_grid_rates(_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    top = g.cell_table.iloc[0]
    assert top["n_events"] == 3
    assert top["annual_rate"] == round(3 / g.catalog_years, 4)


def test_cell_table_sorted_descending_by_rate():
    g = spatial_grid_rates(_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    rates = g.cell_table["annual_rate"].to_numpy()
    assert np.all(np.diff(rates) <= 0)


def test_catalog_years_spans_full_dataframe():
    g = spatial_grid_rates(_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    assert 9.9 < g.catalog_years < 10.1  # 2000 → 2010


def test_grid_bins_cover_region_exactly_without_float_drift():
    """d=0.1'de np.arange kayması enlemde fazladan (bölge-dışı) bir satır üretiyordu."""
    region = Region("tr", min_lat=35.5, max_lat=42.5, min_lon=25.5, max_lon=45.0)
    grid = spatial_grid_rates(_catalog(), region, cell_deg=0.1, min_mag=4.0)
    assert grid.counts.shape == (70, 195)
    assert grid.lat_edges[-1] == pytest.approx(42.5)
    assert grid.lon_edges[-1] == pytest.approx(45.0)


def test_non_divisible_cell_still_covers_the_max_edge():
    grid = spatial_grid_rates(_catalog(), REGION, cell_deg=0.3, min_mag=4.0)
    assert grid.lat_edges[-1] >= REGION.max_lat
    assert grid.lat_edges[-1] - REGION.max_lat < 0.3


@pytest.mark.parametrize("n_rows", [0, 1])
def test_zero_span_catalog_raises_instead_of_infinite_rate(n_rows):
    df = _catalog().iloc[:n_rows]
    with pytest.raises(ValueError, match="süresi"):
        spatial_grid_rates(df, REGION, cell_deg=0.5, min_mag=4.0)


def test_no_events_above_min_mag_gives_empty_but_typed_cell_table():
    grid = spatial_grid_rates(_catalog(), REGION, cell_deg=0.5, min_mag=9.0)
    assert grid.cell_table.empty
    assert list(grid.cell_table.columns) == ["lat", "lon", "n_events", "annual_rate"]
    assert grid.counts.sum() == 0
