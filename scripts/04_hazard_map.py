"""Track A — uzaysal sismisite oran (hazard) haritası üretir.

Üretir:
  - reports/figures/06_hazard_rate_map.png
  - data/processed/turkey_hazard_cells.csv  (en riskli hücreler)

Kullanım:
    ./.venv/bin/python scripts/04_hazard_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.hazard import spatial_grid_rates  # noqa: E402
from eyequake.config import FIGURES_DIR, PROCESSED_DIR, TURKEY  # noqa: E402

CELL_DEG = 0.25
MIN_MAG = 4.0


def main() -> int:
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])
    grid = spatial_grid_rates(df, TURKEY, cell_deg=CELL_DEG, min_mag=MIN_MAG)

    # log10 oran (sıfır hücreleri maskele).
    rate = grid.annual_rate
    masked = np.ma.masked_where(rate <= 0, rate)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    mesh = ax.pcolormesh(
        grid.lon_edges, grid.lat_edges, np.ma.log10(masked),
        cmap="inferno", shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label="log₁₀(yıllık M≥%.1f oranı)" % MIN_MAG)

    # Büyük depremleri (M≥6) işaretle.
    big = df[df["mag"] >= 6.0]
    ax.scatter(big["longitude"], big["latitude"], s=70, marker="*",
               edgecolors="cyan", facecolors="none", linewidths=1.2, label="M≥6.0")

    ax.set(
        xlabel="Boylam", ylabel="Enlem",
        title=f"Sismisite oran (hazard) alanı — {CELL_DEG}° ızgara, "
              f"{grid.catalog_years:.1f} yıl",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_fig = FIGURES_DIR / "06_hazard_rate_map.png"
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)

    out_csv = PROCESSED_DIR / "turkey_hazard_cells.csv"
    grid.cell_table.to_csv(out_csv, index=False)

    print(f"Katalog süresi: {grid.catalog_years:.1f} yıl | hücre: {CELL_DEG}°")
    print(f"Dolu hücre sayısı: {len(grid.cell_table)}")
    print("\n=== EN YÜKSEK ORANLI 12 HÜCRE (M≥%.1f) ===" % MIN_MAG)
    print(grid.cell_table.head(12).to_string(index=False))
    print(f"\nYazıldı:\n  {out_fig}\n  {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
