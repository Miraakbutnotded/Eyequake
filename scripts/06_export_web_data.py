"""Ön yüz için statik GeoJSON verisini web/data/ altına yazar.

Kullanım:
    ./.venv/bin/python scripts/06_export_web_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.config import PROCESSED_DIR, TURKEY, WEB_DATA_DIR  # noqa: E402
from eyequake.web.export import (  # noqa: E402
    build_cell_index,
    cells_to_geojson,
    events_to_geojson,
)

CELL_DEG = 0.25
MIN_MAG = 4.0


def main() -> int:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])

    cells = build_cell_index(df, TURKEY, cell_deg=CELL_DEG, min_mag=MIN_MAG)
    hazard_gj = cells_to_geojson(cells, cell_deg=CELL_DEG)
    events_gj = events_to_geojson(df, min_mag=6.0)

    (WEB_DATA_DIR / "hazard_cells.geojson").write_text(
        json.dumps(hazard_gj), encoding="utf-8"
    )
    (WEB_DATA_DIR / "big_events.geojson").write_text(
        json.dumps(events_gj), encoding="utf-8"
    )
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "USGS FDSN Event API",
        "region": TURKEY.name,
        "cell_deg": CELL_DEG,
        "min_mag": MIN_MAG,
        "n_cells": len(cells),
        "n_big_events": len(events_gj["features"]),
        "disclaimer": (
            "Göreli sismik tehlike indeksi (0–100), geçmiş aktiviteye dayalı. "
            "Olasılık veya deprem tahmini DEĞİLDİR."
        ),
    }
    (WEB_DATA_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Hücre: {len(cells)} | büyük olay: {meta['n_big_events']}")
    print("En riskli 5 hücre:")
    print(cells[["lat", "lon", "risk_index", "annual_rate", "max_mag", "n_recent"]]
          .head(5).to_string(index=False))
    print(f"\nYazıldı: {WEB_DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
