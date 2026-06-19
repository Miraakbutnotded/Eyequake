"""Toplu (batch) yükseklik çekme + cache — open-elevation POST API.

Anahtarsız, global. POST gövdesinde tek istekte yüzlerce nokta gönderilir
(open-meteo'nun koordinat-ağırlıklı saatlik kotasına takılmamak için). Eğim
hesabı için grid hücresi komşularının yüksekliği gerekir. Sonuç diske cache'lenir
(incremental — rerun kaldığı yerden devam eder).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OPEN_ELEVATION = "https://api.open-elevation.com/api/v1/lookup"
_BATCH = 500  # POST gövdesinde tek istekte nokta sayısı


def _key(lat: float, lon: float) -> str:
    return f"{lat:.4f},{lon:.4f}"


def fetch_elevations(
    points: list[tuple[float, float]],
    cache_path: Path | None = None,
    pause_seconds: float = 1.0,
) -> dict[str, float]:
    """(lat, lon) noktaları için yükseklik (m): {"lat,lon": elev}.

    cache_path verilirse önce oradan okur, yalnızca eksik noktaları çeker, her
    chunk sonrası geri yazar (kesintide ilerleme korunur).
    """
    cache: dict[str, float] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    uniq = list({(la, lo) for la, lo in points})
    missing = [(la, lo) for la, lo in uniq if _key(la, lo) not in cache]

    def _save() -> None:
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))

    if missing:
        print(f"  {len(missing)} eksik nokta çekilecek ({len(cache)} cache'li)")
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "EyeQuake-research-prototype/0.1",
                "Content-Type": "application/json",
            })
            for i in range(0, len(missing), _BATCH):
                chunk = missing[i : i + _BATCH]
                _fetch_chunk_with_retry(session, chunk, cache)
                _save()
                print(f"  yükseklik {min(i + _BATCH, len(missing))}/{len(missing)}")
                time.sleep(pause_seconds)

    return {_key(la, lo): cache[_key(la, lo)] for la, lo in points}


def _fetch_chunk_with_retry(session, chunk, cache, max_tries: int = 5) -> None:
    """Tek chunk'ı POST eder; 429/5xx/ağ hatasında exponential backoff ile yeniden dener."""
    body = {"locations": [{"latitude": la, "longitude": lo} for la, lo in chunk]}
    for attempt in range(max_tries):
        try:
            resp = session.post(OPEN_ELEVATION, json=body, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            results = resp.json()["results"]
            for (la, lo), r in zip(chunk, results):
                cache[_key(la, lo)] = float(r["elevation"])
            return
        except (requests.RequestException, KeyError, ValueError) as exc:
            if attempt == max_tries - 1:
                raise RuntimeError(
                    f"Yükseklik çekilemedi ({max_tries} deneme): {exc}. "
                    "Cache'lenenler korundu; scripti tekrar çalıştır."
                ) from exc
            wait = 5 * (2**attempt)  # 5, 10, 20, 40 s
            print(f"    {exc} → {wait}s bekle, yeniden dene ({attempt + 1}/{max_tries})")
            time.sleep(wait)
