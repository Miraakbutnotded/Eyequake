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

from eyequake.analysis.seismology import magnitude_of_completeness  # noqa: E402
from eyequake.config import PROCESSED_DIR, TURKEY, WEB_DATA_DIR  # noqa: E402
from eyequake.forecast.spatial_skill import evaluate_surface_skill  # noqa: E402
from eyequake.web.export import (  # noqa: E402
    build_cell_index,
    cells_to_geojson,
    events_to_geojson,
)

# AFAD kataloğu M≥3.0'da tam → daha yoğun, daha bilgilendirici risk haritası.
CELL_DEG = 0.25
MIN_MAG = 3.0


def main() -> int:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])

    cells = build_cell_index(df, TURKEY, cell_deg=CELL_DEG, min_mag=MIN_MAG)
    hazard_gj = cells_to_geojson(cells, cell_deg=CELL_DEG)
    events_gj = events_to_geojson(df, min_mag=6.0)

    # Mc (magnitude of completeness): MIN_MAG bunun üstünde olmalı, aksi halde
    # tehlike yüzeyi sismisiteyi değil katalog raporlama yanlılığını modeller.
    mc = magnitude_of_completeness(df["mag"].to_numpy(dtype=float))

    # Servis edilen yüzeyin ölçülmüş zaman-dışı (out-of-time) beceri provenance'ı:
    # eğitim penceresinden kurulan yüzey, GELECEK depremleri alan-uniform (Poisson)
    # baseline'a göre yoğunlaştırıyor mu? area_skill_score > 0 → baseline'ı yeniyor.
    # SERVİS PARAMETRELERİNDE (MIN_MAG/CELL_DEG) hesaplanır; seed=0 deterministiktir.
    # area_skill_score_ci: bootstrap %95 GA — tek-bölünme gürültüsünü niceler.
    # Tanımlayıcı bir beceri ölçüsüdür — deprem tahmini DEĞİLDİR.
    skill = evaluate_surface_skill(df, TURKEY, min_mag=MIN_MAG, cell_deg=CELL_DEG, n_boot=1000, seed=0)

    (WEB_DATA_DIR / "hazard_cells.geojson").write_text(
        json.dumps(hazard_gj), encoding="utf-8"
    )
    (WEB_DATA_DIR / "big_events.geojson").write_text(
        json.dumps(events_gj), encoding="utf-8"
    )
    # Kaynak adını işlenmiş özetten oku (USGS ya da AFAD — son çekime bağlı).
    summary_path = PROCESSED_DIR / f"{TURKEY.name}_summary.json"
    source = "AFAD"
    if summary_path.exists():
        source = json.loads(summary_path.read_text()).get("source", "AFAD")

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": source,
        "region": TURKEY.name,
        "cell_deg": CELL_DEG,
        "min_mag": MIN_MAG,
        "mc": mc,
        "area_skill_score": skill["area_skill_score"],
        "area_skill_score_ci": skill["area_skill_score_ci"],
        "gain_top25": skill["gain_top25"],
        "n_test_events": skill["n_test_events"],
        "n_cells": len(cells),
        "n_big_events": len(events_gj["features"]),
        "disclaimer": (
            "Göreli sismik tehlike indeksi (0–100), geçmiş aktiviteye dayalı. "
            "Olasılık veya deprem tahmini DEĞİLDİR."
        ),
        "use_restriction": (
            "Göreli triage / portföy sıralama / accumulation control. "
            "Absolute loss'a kalibre DEĞİL — premium rating için doğrudan girdi DEĞİLDİR."
        ),
    }
    (WEB_DATA_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ci = skill["area_skill_score_ci"]
    ci_str = f"[{ci[0]}, {ci[1]}]" if ci else "n/a"
    print(
        f"Hücre: {len(cells)} | büyük olay: {meta['n_big_events']} | "
        f"area_skill_score: {skill['area_skill_score']} 95%CI={ci_str} "
        f"(gain_top25 {skill['gain_top25']}, n_test {skill['n_test_events']})"
    )
    print("En riskli 5 hücre:")
    print(cells[["lat", "lon", "risk_index", "annual_rate", "max_mag", "n_recent"]]
          .head(5).to_string(index=False))
    print(f"\nYazıldı: {WEB_DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
