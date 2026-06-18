"""Proje geneli sabitler, bölge tanımları ve dosya yolları."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Yollar ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
WEB_DIR = PROJECT_ROOT / "web"
WEB_DATA_DIR = WEB_DIR / "data"


@dataclass(frozen=True)
class Region:
    """Coğrafi bounding box (WGS84)."""

    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def as_params(self) -> dict[str, float]:
        """FDSN sorgu parametreleri."""
        return {
            "minlatitude": self.min_lat,
            "maxlatitude": self.max_lat,
            "minlongitude": self.min_lon,
            "maxlongitude": self.max_lon,
        }


# Türkiye + yakın çevre. Ana fay sistemlerini kapsar:
# Kuzey Anadolu Fayı (KAF) ve Doğu Anadolu Fayı (DAF / 6 Şubat bölgesi).
TURKEY = Region(
    name="turkey",
    min_lat=35.5,
    max_lat=42.5,
    min_lon=25.5,
    max_lon=45.0,
)

# Ana fay sistemleri — kaba dikdörtgen sınırlar (gerçek fay izi değil; bölge-bazlı
# ETAS için sismik olarak bağımsız alt-katalogları ayırmaya yeter).
FAULT_ZONES = [
    Region(name="KAF (Kuzey Anadolu)", min_lat=39.5, max_lat=41.5, min_lon=26.0, max_lon=41.0),
    Region(name="DAF (Doğu Anadolu / 2023)", min_lat=36.0, max_lat=39.0, min_lon=35.5, max_lon=40.0),
    Region(name="Ege / Helen yayı", min_lat=36.0, max_lat=39.5, min_lon=25.5, max_lon=28.5),
    Region(name="Van", min_lat=37.5, max_lat=39.5, min_lon=42.0, max_lon=45.0),
]

# --- Veri kaynağı ---
USGS_FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# --- Katalog varsayılanları ---
CATALOG_START_YEAR = 1990
# Türkiye için ~2.5 altı kayıt, tarihsel olarak tamamlanmamış (magnitude of
# completeness). Daha düşük eşik analizleri sapmalı yapar.
DEFAULT_MIN_MAGNITUDE = 2.5
