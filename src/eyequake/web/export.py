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


def cell_bins(lat, lon, region: Region, cell_deg: float):
    """(lat, lon) → bölge köşesine göre tamsayı ızgara indeksleri (floor bölme).

    Üretici (`build_cell_index`) ve tüketici (`forecast.spatial_skill.
    evaluate_surface_skill`) AYNI hücre tanımını paylaşsın diye tek kaynak;
    aksi halde ızgaraları sessizce ıraksayıp beceri ölçümünü bozabilirdi.
    """
    lat_bin = np.floor((lat - region.min_lat) / cell_deg).astype(int)
    lon_bin = np.floor((lon - region.min_lon) / cell_deg).astype(int)
    return lat_bin, lon_bin


def build_cell_index(
    df: pd.DataFrame, region: Region, *, cell_deg: float = 0.25,
    min_mag: float, recent_years: int = 10,
) -> pd.DataFrame:
    """Hücre başına oran, maks büyüklük, güncel aktivite ve göreli risk indeksi."""
    times = _to_utc(df["time"])
    catalog_years = (times.max() - times.min()).total_seconds() / _SECONDS_PER_YEAR
    recent_cut = times.max() - pd.Timedelta(days=recent_years * 365.25)

    sub = df[df["mag"] >= min_mag].copy()
    sub["time"] = _to_utc(sub["time"])
    sub["lat_bin"], sub["lon_bin"] = cell_bins(sub["latitude"], sub["longitude"], region, cell_deg)
    if "depth" in sub.columns:
        sub["depth"] = pd.to_numeric(sub["depth"], errors="coerce").fillna(999.0)
    else:
        sub["depth"] = pd.Series(999.0, index=sub.index)
    sub["energy_proxy"] = np.power(10.0, 1.5 * sub["mag"])
    sub["is_shallow"] = sub["depth"] <= 15.0

    agg = (
        sub.groupby(["lat_bin", "lon_bin"])
        .agg(
            n_events=("mag", "size"),
            max_mag=("mag", "max"),
            mean_depth=("depth", "mean"),
            shallow_fraction=("is_shallow", "mean"),
            energy_sum=("energy_proxy", "sum"),
            last_event=("time", "max"),
        )
        .reset_index()
    )
    recent = (
        sub[sub["time"] >= recent_cut]
        .groupby(["lat_bin", "lon_bin"]).size().rename("n_recent").reset_index()
    )
    cells = agg.merge(recent, on=["lat_bin", "lon_bin"], how="left")
    cells["n_recent"] = cells["n_recent"].fillna(0).astype(int)

    cells["annual_rate"] = (cells["n_events"] / catalog_years).round(4)
    cells["recent_rate"] = (cells["n_recent"] / recent_years).round(4)
    cells["recency_days"] = (
        (times.max() - cells["last_event"]).dt.total_seconds() / 86_400.0
    ).round(1)
    cells["energy_index"] = np.log10(cells["energy_sum"]).round(2)
    cells["mean_depth"] = cells["mean_depth"].round(1)
    cells["shallow_fraction"] = cells["shallow_fraction"].round(3)
    cells["lat"] = region.min_lat + (cells["lat_bin"] + 0.5) * cell_deg
    cells["lon"] = region.min_lon + (cells["lon_bin"] + 0.5) * cell_deg

    # v2 AFAD multi-signal risk: tanımlayıcı hücre sıralaması, deprem tahmini değil.
    cells["risk_rate_component"] = cells["annual_rate"].rank(pct=True)
    cells["risk_magnitude_component"] = cells["max_mag"].rank(pct=True)
    cells["risk_recent_component"] = cells["recent_rate"].rank(pct=True)
    cells["risk_energy_component"] = cells["energy_index"].rank(pct=True)
    cells["risk_recency_component"] = (-cells["recency_days"]).rank(pct=True)
    cells["risk_shallow_component"] = cells["shallow_fraction"].rank(pct=True)
    cells["risk_index"] = (
        100
        * (
            0.22 * cells["risk_rate_component"]
            + 0.18 * cells["risk_magnitude_component"]
            + 0.18 * cells["risk_recent_component"]
            + 0.18 * cells["risk_energy_component"]
            + 0.16 * cells["risk_recency_component"]
            + 0.08 * cells["risk_shallow_component"]
        )
    ).round(0).astype(int)
    cells["risk_model"] = "v2_afad_multi_signal"

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
                "risk_model": str(r.get("risk_model", "v2_afad_multi_signal")),
                "risk_index": int(r["risk_index"]),
                "annual_rate": float(r["annual_rate"]),
                "recent_rate": float(r.get("recent_rate", 0.0)),
                "recency_days": float(r.get("recency_days", 0.0)),
                "energy_index": float(r.get("energy_index", 0.0)),
                "mean_depth": float(r.get("mean_depth", 0.0)),
                "shallow_fraction": float(r.get("shallow_fraction", 0.0)),
                "risk_rate_component": round(float(r.get("risk_rate_component", 0.0)), 3),
                "risk_magnitude_component": round(float(r.get("risk_magnitude_component", 0.0)), 3),
                "risk_recent_component": round(float(r.get("risk_recent_component", 0.0)), 3),
                "risk_energy_component": round(float(r.get("risk_energy_component", 0.0)), 3),
                "risk_recency_component": round(float(r.get("risk_recency_component", 0.0)), 3),
                "risk_shallow_component": round(float(r.get("risk_shallow_component", 0.0)), 3),
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
