"""Ham FDSN kataloğunu analiz-hazır forma normalize etme ve diskten okuma."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROCESSED_DIR, TURKEY

# Analizde kullanılan kolonlar.
_KEEP = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place"]


def catalog_path() -> Path:
    """Kanonik temizlenmiş katalog dosyasının yolu."""
    return PROCESSED_DIR / f"{TURKEY.name}_catalog.csv"


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    """Kanonik katalogu okur. Dosya yoksa hangi fetcher'ın koşacağını söyler.

    Kataloğu 07_fetch_afad.py (kanonik) veya 01_fetch_catalog.py (USGS yedeği) yazar;
    ikisi de aynı şemayı üretir, bu yüzden downstream kaynağı bilmek zorunda değildir.
    """
    path = path or catalog_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Katalog yok: {path}\n"
            "Önce bir fetcher çalıştır: scripts/07_fetch_afad.py (kanonik) "
            "veya scripts/01_fetch_catalog.py (USGS yedeği)."
        )
    return pd.read_csv(path, parse_dates=["time"])


def clean_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    """Ham katalogu temizler ve türetilmiş zaman alanları ekler.

    Adımlar:
      - zamanı UTC datetime'a çevir, kronolojik sırala
      - konum/büyüklük eksik satırları at
      - yinelenen olayları (id) düşür
      - yıl ve ardışık-olaylar-arası süre (gün) türet

    Returns:
        Temizlenmiş katalog (zamana göre artan sıralı).
    """
    df = raw.loc[:, [c for c in _KEEP if c in raw.columns]].copy()

    # format="ISO8601": değişken hassasiyetli ISO tarihleri (AFAD bazı kayıtlarda
    # fractional saniye taşır) tutarlı parse eder. Sabit format çıkarımı, eşleşmeyen
    # satırları sessizce NaT'ye çevirip veri kaybına yol açardı.
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601", errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"])
    df = df.drop_duplicates(subset="id")
    df = df.sort_values("time").reset_index(drop=True)

    df["year"] = df["time"].dt.year
    # Ardışık olaylar arası süre (gün) — sismisite oranı analizleri için.
    df["interevent_days"] = df["time"].diff().dt.total_seconds() / 86_400.0

    return df


def completeness_summary(df: pd.DataFrame) -> dict[str, object]:
    """Katalog hakkında hızlı kalite/kapsam özeti."""
    return {
        "n_events": int(len(df)),
        "date_min": df["time"].min().isoformat() if len(df) else None,
        "date_max": df["time"].max().isoformat() if len(df) else None,
        "mag_min": float(df["mag"].min()) if len(df) else None,
        "mag_max": float(df["mag"].max()) if len(df) else None,
        "depth_median_km": float(df["depth"].median()) if len(df) else None,
    }
