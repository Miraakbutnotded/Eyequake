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


def _fetch_year(session, url: str, timeout: int, max_tries: int, backoff_seconds: float):
    """Tek yılı çeker; ağ/HTTP/JSON hatasında üstel backoff ile yeniden dener."""
    for attempt in range(max_tries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            records = resp.json() if resp.text.strip() else []
            if not isinstance(records, list):
                raise ValueError(f"beklenmeyen AFAD yanıtı: {type(records).__name__}")
            return _normalize(records)
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_tries - 1:
                raise
            wait = backoff_seconds * (2**attempt)
            print(f"    {exc} -> {wait:g}s bekle, yeniden dene ({attempt + 1}/{max_tries})")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_catalog_afad(
    region: Region,
    start_year: int,
    end_year: int,
    min_magnitude: float,
    pause_seconds: float = 0.4,
    timeout: int = 120,
    max_tries: int = 4,
    backoff_seconds: float = 5.0,
    allow_gaps: bool = False,
    min_events_per_year: int = 1,
) -> pd.DataFrame:
    """AFAD kataloğunu yıllık paging ile çeker, ortak şemaya normalize eder.

    Bir yıl tüm denemelerden sonra hâlâ çekilemezse varsayılan olarak RuntimeError
    (fail-closed): eksik bir yıl kanonik katalogda sessiz bir boşluk bırakır, tüm
    yıllık oranları aşağı çeker ve zaman-dışı beceri ölçümünü bozar. Boşluklu bir
    katalog bilerek isteniyorsa `allow_gaps=True`.

    `min_events_per_year` altında olay dönen yıl da eksik sayılır: HTTP 200 + boş
    liste bir "başarı" gibi görünür ama Türkiye'de M≥2.5'te olaysız bir yıl yoktur.
    (2026-10'da AFAD apiv2 yalnızca son ~11 günü döndürmeye başladı; 1990-2025 her
    yıl 0 olay geldi ve 45 olaylık bir "kanonik katalog" yazıldı.) Küçük bölge /
    yüksek eşik için 0 geç.
    """
    frames: list[pd.DataFrame] = []
    failed: dict[int, str] = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": "EyeQuake-research-prototype/0.1"})
        for year in range(start_year, end_year + 1):
            url = _build_url(
                f"{year}-01-01 00:00:00", f"{year + 1}-01-01 00:00:00",
                region, min_magnitude,
            )
            try:
                frame = _fetch_year(session, url, timeout, max_tries, backoff_seconds)
                frames.append(frame)
                print(f"  {year}: {len(frame):>7} olay")
                if len(frame) < min_events_per_year:
                    failed[year] = f"yalnızca {len(frame)} olay (< {min_events_per_year})"
            except (requests.RequestException, ValueError) as exc:
                failed[year] = str(exc)
                print(f"  {year}: HATA — {exc}")
            time.sleep(pause_seconds)

    if not frames:
        raise RuntimeError("AFAD'dan hiç veri çekilemedi.")
    if failed and not allow_gaps:
        years = ", ".join(str(y) for y in sorted(failed))
        raise RuntimeError(
            f"AFAD kataloğu eksik: {years} yılları çekilemedi ya da boş döndü. "
            "Boşluklu katalog yıllık oranları sessizce düşürür; tekrar çalıştır "
            "(ya da bilerek allow_gaps=True)."
        )

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset="id").reset_index(drop=True)
    return catalog
