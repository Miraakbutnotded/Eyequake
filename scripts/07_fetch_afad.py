"""AFAD (resmi ulusal ağ) kataloğunu indirir → kanonik katalog olur.

USGS'in tamlık sorununu çözer. Çıktı processed/turkey_catalog.csv'yi günceller,
böylece downstream scriptler (02-06) otomatik AFAD verisini kullanır.

Kullanım:
    ./.venv/bin/python scripts/07_fetch_afad.py
    ./.venv/bin/python scripts/07_fetch_afad.py --start-year 2010 --min-mag 2.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from eyequake.config import CATALOG_START_YEAR, PROCESSED_DIR, RAW_DIR, TURKEY
from eyequake.data.clean import catalog_path, clean_catalog, completeness_summary
from eyequake.data.fetch_afad import fetch_catalog_afad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AFAD deprem kataloğunu indir.")
    p.add_argument("--start-year", type=int, default=CATALOG_START_YEAR)
    p.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    p.add_argument("--min-mag", type=float, default=2.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"AFAD → {TURKEY.name} | {args.start_year}-{args.end_year} | M≥{args.min_mag}")
    raw = fetch_catalog_afad(
        region=TURKEY,
        start_year=args.start_year,
        end_year=args.end_year,
        min_magnitude=args.min_mag,
    )
    raw.to_csv(RAW_DIR / f"{TURKEY.name}_catalog_afad_raw.csv", index=False)

    clean = clean_catalog(raw)
    clean.to_csv(catalog_path(), index=False)  # kanonik

    summary = completeness_summary(clean)
    summary["source"] = "AFAD"
    (PROCESSED_DIR / f"{TURKEY.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("\n--- ÖZET (AFAD) ---")
    print(f"Ham olay: {len(raw)} | temiz olay: {summary['n_events']}")
    print(f"Tarih: {summary['date_min']} → {summary['date_max']}")
    print(f"Büyüklük: {summary['mag_min']} – {summary['mag_max']}")
    print("\nKanonik katalog güncellendi. Şimdi 02-06 scriptlerini AFAD ile çalıştır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
