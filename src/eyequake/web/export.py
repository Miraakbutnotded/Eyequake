"""Ön yüz için statik GeoJSON üretimi.

Göreli sismik tehlike indeksi: Türkiye genelinde geçmiş aktiviteye dayalı
yüzdelik (0–100). Bu bir OLASILIK ya da TAHMİN DEĞİL — tanımlayıcı, göreli bir
göstergedir. Olasılık dili kullanmaktan kasıtla kaçınılır (dürüst konumlandırma).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from ..config import Region

_SECONDS_PER_YEAR = 365.25 * 86_400.0


def _to_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, utc=True, format="ISO8601")


def build_cell_index(
    df: pd.DataFrame, region: Region, cell_deg: float = 0.25,
    min_mag: float = 4.0, recent_years: int = 10,
) -> pd.DataFrame:
    """Hücre başına oran, maks büyüklük, güncel aktivite ve göreli risk indeksi."""
    times = _to_utc(df["time"])
    catalog_years = (times.max() - times.min()).total_seconds() / _SECONDS_PER_YEAR
    recent_cut = times.max() - pd.Timedelta(days=recent_years * 365.25)

    sub = df[df["mag"] >= min_mag].copy()
    sub["time"] = _to_utc(sub["time"])
    sub["lat_bin"] = np.floor((sub["latitude"] - region.min_lat) / cell_deg).astype(int)
    sub["lon_bin"] = np.floor((sub["longitude"] - region.min_lon) / cell_deg).astype(int)

    agg = (
        sub.groupby(["lat_bin", "lon_bin"])
        .agg(n_events=("mag", "size"), max_mag=("mag", "max"))
        .reset_index()
    )
    recent = (
        sub[sub["time"] >= recent_cut]
        .groupby(["lat_bin", "lon_bin"]).size().rename("n_recent").reset_index()
    )
    cells = agg.merge(recent, on=["lat_bin", "lon_bin"], how="left")
    cells["n_recent"] = cells["n_recent"].fillna(0).astype(int)

    cells["annual_rate"] = (cells["n_events"] / catalog_years).round(4)
    cells["lat"] = region.min_lat + (cells["lat_bin"] + 0.5) * cell_deg
    cells["lon"] = region.min_lon + (cells["lon_bin"] + 0.5) * cell_deg

    # Göreli indeks: oran (log) ve maks büyüklük yüzdeliklerinin ağırlıklı bileşimi.
    rate_pct = cells["annual_rate"].rank(pct=True)
    mag_pct = cells["max_mag"].rank(pct=True)
    cells["risk_index"] = (100 * (0.6 * rate_pct + 0.4 * mag_pct)).round(0).astype(int)

    return cells.sort_values("risk_index", ascending=False).reset_index(drop=True)


def cells_to_geojson(cells: pd.DataFrame, cell_deg: float = 0.25) -> dict:
    """Hücreleri kare poligon FeatureCollection'a çevirir."""
    features = []
    half = cell_deg / 2.0
    for _, r in cells.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        poly = [[
            [lon - half, lat - half], [lon + half, lat - half],
            [lon + half, lat + half], [lon - half, lat + half],
            [lon - half, lat - half],
        ]]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": poly},
            "properties": {
                "risk_index": int(r["risk_index"]),
                "annual_rate": float(r["annual_rate"]),
                "max_mag": round(float(r["max_mag"]), 1),
                "n_events": int(r["n_events"]),
                "n_recent": int(r["n_recent"]),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def events_to_geojson(df: pd.DataFrame, min_mag: float = 6.0) -> dict:
    """Büyük olayları (referans/bağlam için) nokta FeatureCollection'a çevirir."""
    big = df[df["mag"] >= min_mag].copy()
    big["time"] = _to_utc(big["time"])
    features = []
    for _, r in big.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["longitude"]), float(r["latitude"])],
            },
            "properties": {
                "mag": round(float(r["mag"]), 1),
                "date": r["time"].strftime("%Y-%m-%d"),
                "place": str(r.get("place", "")),
            },
        })
    return {"type": "FeatureCollection", "features": features}
