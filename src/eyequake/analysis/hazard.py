"""Track A — uzaysal sismisite oran (hazard) alanı.

PSHA'nın (Probabilistic Seismic Hazard Assessment) çekirdek girdisi olan yıllık
deprem oranı alanını (a-value field) ızgara üzerinde hesaplar. Olasılıksızdır:
"bu hücre yılda kaç M≥M0 deprem üretir" — savunulabilir, standart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Region

_SECONDS_PER_YEAR = 365.25 * 86_400.0


@dataclass(frozen=True)
class HazardGrid:
    """Izgara tabanlı sismisite oran alanı."""

    lon_edges: np.ndarray
    lat_edges: np.ndarray
    counts: np.ndarray  # [n_lat, n_lon] olay sayısı
    annual_rate: np.ndarray  # [n_lat, n_lon] olay/yıl
    catalog_years: float
    cell_table: pd.DataFrame  # boş olmayan hücreler (tidy)


def _to_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, utc=True, format="ISO8601")


def spatial_grid_rates(
    df: pd.DataFrame, region: Region, cell_deg: float = 0.25, min_mag: float = 4.0
) -> HazardGrid:
    """Bölgeyi cell_deg ızgarasına böler, hücre başına yıllık oran hesaplar.

    Oran paydası tüm katalog süresidir (tutarlı normalizasyon).
    """
    times = _to_utc(df["time"])
    catalog_years = (times.max() - times.min()).total_seconds() / _SECONDS_PER_YEAR

    sub = df[df["mag"] >= min_mag]
    lon_edges = np.arange(region.min_lon, region.max_lon + cell_deg, cell_deg)
    lat_edges = np.arange(region.min_lat, region.max_lat + cell_deg, cell_deg)

    counts, _, _ = np.histogram2d(
        sub["latitude"].to_numpy(),
        sub["longitude"].to_numpy(),
        bins=[lat_edges, lon_edges],
    )
    annual_rate = counts / catalog_years

    # Tidy tablo: boş olmayan hücreler (en riskliyi sıralamak için).
    lat_centers = lat_edges[:-1] + cell_deg / 2.0
    lon_centers = lon_edges[:-1] + cell_deg / 2.0
    rows = []
    for i, lat in enumerate(lat_centers):
        for j, lon in enumerate(lon_centers):
            if counts[i, j] > 0:
                rows.append(
                    {
                        "lat": round(float(lat), 3),
                        "lon": round(float(lon), 3),
                        "n_events": int(counts[i, j]),
                        "annual_rate": round(float(annual_rate[i, j]), 4),
                    }
                )
    cell_table = pd.DataFrame(rows).sort_values("annual_rate", ascending=False)

    return HazardGrid(
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        counts=counts,
        annual_rate=annual_rate,
        catalog_years=catalog_years,
        cell_table=cell_table.reset_index(drop=True),
    )
