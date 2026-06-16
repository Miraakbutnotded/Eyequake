"""USGS FDSN Event API'sinden deprem kataloğu çekme.

FDSN `query` endpoint'i tek istekte en fazla 20.000 olay döndürür. Uzun zaman
aralıklarında bu sınırı aşmamak için yıllık pencerelerle paging yaparız.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import pandas as pd
import requests

from ..config import USGS_FDSN_URL, Region

# FDSN CSV çıktısındaki ham kolon adları (kullandıklarımız).
_RAW_COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "id", "place"]


@dataclass(frozen=True)
class FetchResult:
    """Çekim sonucu + metadata (gözlemlenebilirlik için)."""

    catalog: pd.DataFrame
    n_events: int
    n_windows: int
    n_failed_windows: int


def _fetch_window(
    region: Region,
    start: str,
    end: str,
    min_magnitude: float,
    session: requests.Session,
    timeout: int = 90,
) -> pd.DataFrame:
    """Tek bir [start, end) zaman penceresini çeker. Boşsa boş DataFrame döner."""
    params = {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
        **region.as_params(),
    }
    resp = session.get(USGS_FDSN_URL, params=params, timeout=timeout)

    # FDSN, olay yokken 204 No Content döndürür.
    if resp.status_code == 204:
        return pd.DataFrame(columns=_RAW_COLUMNS)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text:
        return pd.DataFrame(columns=_RAW_COLUMNS)

    df = pd.read_csv(io.StringIO(resp.text))
    if len(df) >= 20_000:
        raise RuntimeError(
            f"Pencere {start}–{end} 20.000 olay sınırına ulaştı; daha küçük "
            "pencere (aylık) gerekiyor."
        )
    return df


def fetch_catalog(
    region: Region,
    start_year: int,
    end_year: int,
    min_magnitude: float,
    pause_seconds: float = 0.5,
) -> FetchResult:
    """Bölge için [start_year, end_year] kataloğunu yıllık paging ile çeker.

    Args:
        region: Coğrafi bounding box.
        start_year: Dahil başlangıç yılı.
        end_year: Dahil bitiş yılı.
        min_magnitude: Alt büyüklük eşiği.
        pause_seconds: API'ye nazik olmak için pencereler arası bekleme.

    Returns:
        FetchResult — birleştirilmiş, id'ye göre tekilleştirilmiş katalog.
    """
    frames: list[pd.DataFrame] = []
    n_windows = 0
    n_failed = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "EyeQuake-research-prototype/0.1"})
        for year in range(start_year, end_year + 1):
            start = f"{year}-01-01T00:00:00"
            end = f"{year + 1}-01-01T00:00:00"
            n_windows += 1
            try:
                frame = _fetch_window(region, start, end, min_magnitude, session)
                frames.append(frame)
                print(f"  {year}: {len(frame):>6} olay")
            except (requests.RequestException, RuntimeError) as exc:
                n_failed += 1
                print(f"  {year}: HATA — {exc}")
            time.sleep(pause_seconds)

    if not frames:
        raise RuntimeError("Hiçbir pencere çekilemedi — ağ bağlantısını kontrol et.")

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset="id").reset_index(drop=True)

    return FetchResult(
        catalog=catalog,
        n_events=len(catalog),
        n_windows=n_windows,
        n_failed_windows=n_failed,
    )
