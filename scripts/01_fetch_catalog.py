"""USGS FDSN'den Türkiye deprem kataloğunu indirir, temizler ve diske yazar.

Kullanım:
    ./.venv/bin/python scripts/01_fetch_catalog.py
    ./.venv/bin/python scripts/01_fetch_catalog.py --start-year 2000 --min-mag 3.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

# src/ layout'unu import yoluna ekle.

from eyequake.config import (
    CATALOG_START_YEAR,
    DEFAULT_MIN_MAGNITUDE,
    PROCESSED_DIR,
    RAW_DIR,
    TURKEY,
)
from eyequake.data.clean import catalog_path, clean_catalog, completeness_summary
from eyequake.data.fetch import fetch_catalog


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Türkiye deprem kataloğunu indir.")
    p.add_argument("--start-year", type=int, default=CATALOG_START_YEAR)
    p.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    p.add_argument("--min-mag", type=float, default=DEFAULT_MIN_MAGNITUDE)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"USGS FDSN → {TURKEY.name} | {args.start_year}-{args.end_year} | "
        f"M≥{args.min_mag}"
    )
    result = fetch_catalog(
        region=TURKEY,
        start_year=args.start_year,
        end_year=args.end_year,
        min_magnitude=args.min_mag,
    )

    raw_path = RAW_DIR / f"{TURKEY.name}_catalog_raw.csv"
    result.catalog.to_csv(raw_path, index=False)

    clean = clean_catalog(result.catalog)
    clean_path = catalog_path()
    clean.to_csv(clean_path, index=False)

    summary = completeness_summary(clean)
    summary_path = PROCESSED_DIR / f"{TURKEY.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n--- ÖZET ---")
    print(f"Çekilen pencere: {result.n_windows} (başarısız: {result.n_failed_windows})")
    print(f"Ham olay: {result.n_events} | temiz olay: {summary['n_events']}")
    print(f"Tarih aralığı: {summary['date_min']} → {summary['date_max']}")
    print(f"Büyüklük: {summary['mag_min']} – {summary['mag_max']}")
    print(f"Medyan derinlik: {summary['depth_median_km']} km")
    print(f"\nYazıldı:\n  {raw_path}\n  {clean_path}\n  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
