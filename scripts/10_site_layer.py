"""Zemin/saha katmanı — hazard_cells.geojson'u site metrikleriyle zenginleştirir.

Her hücre için: yükseklik, eğim, Vs30, NEHRP sınıfı, büyütme, sıvılaşma yatkınlığı
(screening), saha-düzeltilmiş risk. open-meteo yükseklikten türetir.

Kullanım:
    ./.venv/bin/python scripts/10_site_layer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.site import (  # noqa: E402
    amplification,
    compute_slope,
    liquefaction,
    nehrp_class,
    site_adjusted_risk,
    slope_to_vs30,
)
from eyequake.config import RAW_DIR, WEB_DATA_DIR  # noqa: E402
from eyequake.data.elevation import fetch_elevations  # noqa: E402

SLOPE_DELTA = 0.02  # ° — yerel eğim için komşu mesafesi (~2 km)

# (ad, lat, lon, beklenti) — sanity-check
SPOT_CHECKS = [
    ("Adapazarı (1999 sıvılaşma)", 40.78, 30.40, "yüksek amp + sıvılaşma"),
    ("İstanbul Avcılar", 40.98, 28.72, "yüksek amp"),
    ("Çukurova/Adana deltası", 37.00, 35.32, "yüksek sıvılaşma"),
    ("Erzurum yüksek ovası", 39.90, 41.27, "orta/düşük"),
    ("Hakkari dağlık", 37.57, 43.74, "düşük amp/sıvılaşma"),
]


def cell_center(feature: dict) -> tuple[float, float]:
    """Kare poligonun merkezi (lat, lon)."""
    ring = feature["geometry"]["coordinates"][0]
    lon = (ring[0][0] + ring[2][0]) / 2.0
    lat = (ring[0][1] + ring[2][1]) / 2.0
    return round(lat, 4), round(lon, 4)


def main() -> int:
    gj_path = WEB_DATA_DIR / "hazard_cells.geojson"
    gj = json.loads(gj_path.read_text())
    features = gj["features"]
    print(f"Hücre: {len(features)}")

    # Her hücre için merkez + 4 komşu (eğim) noktalarını topla.
    centers, points = [], []
    for f in features:
        lat, lon = cell_center(f)
        centers.append((lat, lon))
        points += [
            (lat, lon),                      # merkez (yükseklik)
            (lat, round(lon + SLOPE_DELTA, 4)),  # doğu
            (lat, round(lon - SLOPE_DELTA, 4)),  # batı
            (round(lat + SLOPE_DELTA, 4), lon),  # kuzey
            (round(lat - SLOPE_DELTA, 4), lon),  # güney
        ]

    print(f"Yükseklik noktası: {len(points)} (cache'li)")
    elev = fetch_elevations(
        points, cache_path=RAW_DIR / "elevation_cache.json", pause_seconds=1.0
    )

    def E(la, lo):
        return elev[f"{la:.4f},{lo:.4f}"]

    counts: dict[str, int] = {}
    for f, (lat, lon) in zip(features, centers):
        e_c = E(lat, lon)
        slope = compute_slope(
            E(lat, round(lon + SLOPE_DELTA, 4)), E(lat, round(lon - SLOPE_DELTA, 4)),
            E(round(lat + SLOPE_DELTA, 4), lon), E(round(lat - SLOPE_DELTA, 4), lon),
            lat, SLOPE_DELTA,
        )
        vs30 = slope_to_vs30(slope)
        cls = nehrp_class(vs30)
        amp = amplification(vs30)
        liq_score, liq_cls = liquefaction(vs30, e_c, slope)
        s_risk = site_adjusted_risk(f["properties"]["risk_index"], amp)

        f["properties"].update({
            "elevation": round(e_c, 1),
            "slope": round(slope, 5),
            "vs30": vs30,
            "site_class": cls,
            "amp": amp,
            "liq_score": liq_score,
            "liq_class": liq_cls,
            "site_risk": s_risk,
        })
        counts[cls] = counts.get(cls, 0) + 1

    gj_path.write_text(json.dumps(gj))

    print("\n=== NEHRP zemin sınıfı dağılımı ===")
    for c in ["E", "D", "C", "B", "A"]:
        if counts.get(c):
            print(f"  {c}: {counts[c]} hücre")

    print("\n=== SANITY-CHECK (bilinen jeoloji) ===")
    for name, lat, lon, beklenti in SPOT_CHECKS:
        nearest = min(centers, key=lambda c: (c[0] - lat) ** 2 + (c[1] - lon) ** 2)
        idx = centers.index(nearest)
        p = features[idx]["properties"]
        print(f"  {name:<32} Vs30={p['vs30']:>3} ({p['site_class']}) "
              f"amp={p['amp']:.2f} sıvılaşma={p['liq_class']:<9} "
              f"site_risk={p['site_risk']:>3}  → beklenti: {beklenti}")

    print(f"\nYazıldı: {gj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
