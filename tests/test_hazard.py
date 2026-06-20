"""hazard.py karakterizasyon testleri (uzaysal grid sayım + oran)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
