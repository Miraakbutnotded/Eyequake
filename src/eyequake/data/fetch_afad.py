"""AFAD (resmi ulusal ağ) deprem kataloğu çekme.

AFAD apiv2, USGS'e göre Türkiye'de çok daha düşük tamlık eşiği (Mc ~2.0) ve çok
daha fazla olay sunar. Çıktı, USGS fetcher ile AYNI şemaya normalize edilir
(id, time, latitude, longitude, depth, mag, magType, place) — böylece tüm
downstream (clean, EDA, hazard, web) değişmeden çalışır.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pandas as pd
import requests

from ..config import Region

AFAD_URL = "https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter"

_SCHEMA = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place"]


def _build_url(start: str, end: str, region: Region, min_magnitude: float) -> str:
    """AFAD'ın beklediği biçimde URL kurar (boşluk %20, ':' korunur)."""
    params = {
        "start": start,
        "end": end,
        "minlat": region.min_lat,
        "maxlat": region.max_lat,
        "minlon": region.min_lon,
        "maxlon": region.max_lon,
        "minmag": min_magnitude,
    }
    query = "&".join(f"{k}={quote(str(v), safe=':-')}" for k, v in params.items())
    return f"{AFAD_URL}?{query}"


def _normalize(records: list[dict]) -> pd.DataFrame:
    """AFAD JSON kayıtlarını ortak şemaya çevirir."""
    if not records:
        return pd.DataFrame(columns=_SCHEMA)
    df = pd.DataFrame(records)
    return pd.DataFrame(
        {
            "id": df["eventID"].astype(str),
            "time": df["date"],  # naive ISO, UTC kabul edilir (clean.py işler)
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "depth": pd.to_numeric(df["depth"], errors="coerce"),
            "mag": pd.to_numeric(df["magnitude"], errors="coerce"),
            "magType": df.get("type"),
            "place": df.get("location"),
        }
    )


def fetch_catalog_afad(
    region: Region,
    start_year: int,
    end_year: int,
    min_magnitude: float,
    pause_seconds: float = 0.4,
    timeout: int = 120,
) -> pd.DataFrame:
    """AFAD kataloğunu yıllık paging ile çeker, ortak şemaya normalize eder."""
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "EyeQuake-research-prototype/0.1"})
        for year in range(start_year, end_year + 1):
            url = _build_url(
                f"{year}-01-01 00:00:00", f"{year + 1}-01-01 00:00:00",
                region, min_magnitude,
            )
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                records = resp.json() if resp.text.strip() else []
                frame = _normalize(records)
                frames.append(frame)
                print(f"  {year}: {len(frame):>7} olay")
            except (requests.RequestException, ValueError) as exc:
                print(f"  {year}: HATA — {exc}")
            time.sleep(pause_seconds)

    if not frames:
        raise RuntimeError("AFAD'dan hiç veri çekilemedi.")

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset="id").reset_index(drop=True)
    return catalog
