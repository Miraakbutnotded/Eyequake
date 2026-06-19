"""Zemin/saha kırılganlık hesapları — Track A'nın site-etkisi katmanı.

Saf, test edilebilir fonksiyonlar:
  - compute_slope        : 4 komşu yükseklikten yerel eğim (m/m)
  - slope_to_vs30        : Wald & Allen (2007) slope-Vs30 (aktif tektonik)
  - nehrp_class          : Vs30 → NEHRP zemin sınıfı (A–E)
  - amplification        : Vs30 → Borcherdt tarzı büyütme faktörü
  - liquefaction         : screening proxy (jeoteknik DEĞİL) → skor + sınıf
  - site_adjusted_risk   : hazard indeksi × büyütme → saha-düzeltilmiş risk

Bilimsel sınır: Vs30 eğimden türetilir (~bölgesel çözünürlük), mikrobölgeleme
yerine geçmez. Sıvılaşma bir tarama göstergesidir; sondaj/yeraltı suyu yok.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Wald & Allen (2007) aktif tektonik slope (m/m) → Vs30 (m/s). Üst sınır dışlayan.
_SLOPE_VS30 = [
    (3.0e-4, 150),   # çok düz → yumuşak (NEHRP E)
    (3.5e-3, 240),
    (1.0e-2, 300),
    (1.8e-2, 360),
    (5.0e-2, 490),
    (1.0e-1, 620),
    (float("inf"), 760),  # dik → kaya (B)
]

_VS30_MIN, _VS30_REF = 150.0, 760.0


@dataclass(frozen=True)
class SiteMetrics:
    elevation: float
    slope: float
    vs30: int
    site_class: str
    amp: float
    liq_score: float
    liq_class: str
    site_risk: int


def compute_slope(e_e: float, e_w: float, e_n: float, e_s: float,
                  lat_deg: float, d_deg: float) -> float:
    """4 komşu yükseklikten (doğu/batı/kuzey/güney) yerel eğim (m/m)."""
    dx_m = d_deg * 111_320.0 * math.cos(math.radians(lat_deg))
    dy_m = d_deg * 110_540.0
    dzdx = (e_e - e_w) / (2.0 * dx_m) if dx_m else 0.0
    dzdy = (e_n - e_s) / (2.0 * dy_m) if dy_m else 0.0
    return math.hypot(dzdx, dzdy)


def slope_to_vs30(slope: float) -> int:
    """Wald-Allen slope-Vs30 (aktif tektonik). Eğim arttıkça Vs30 artar."""
    s = abs(slope)
    for upper, vs30 in _SLOPE_VS30:
        if s < upper:
            return vs30
    return 760


def nehrp_class(vs30: float) -> str:
    if vs30 >= 1500:
        return "A"
    if vs30 >= 760:
        return "B"
    if vs30 >= 360:
        return "C"
    if vs30 >= 180:
        return "D"
    return "E"


def amplification(vs30: float, ref: float = _VS30_REF) -> float:
    """Borcherdt tarzı büyütme: (ref/Vs30)^0.5. Vs30 düştükçe monoton artar."""
    return round(min(3.0, max(0.8, (ref / vs30) ** 0.5)), 2)


def liquefaction(vs30: float, elevation: float, slope: float,
                 annual_precip: float | None = None) -> tuple[float, str]:
    """Screening proxy → (skor 0–1, sınıf). Jeoteknik değerlendirme DEĞİL.

    Yatkınlık ↑: düşük Vs30 (yumuşak), düşük yükseklik (alüvyon ovası), düz eğim,
    (varsa) yüksek yağış. Sondaj/yeraltı suyu verisi içermez.
    """
    s_vs30 = _clip01((_VS30_REF - vs30) / (_VS30_REF - _VS30_MIN))
    s_elev = _clip01((500.0 - elevation) / 500.0)
    s_slope = _clip01((0.02 - abs(slope)) / 0.02)

    if annual_precip is not None:
        s_precip = _clip01(annual_precip / 1000.0)
        score = 0.45 * s_vs30 + 0.25 * s_elev + 0.18 * s_slope + 0.12 * s_precip
    else:
        score = 0.50 * s_vs30 + 0.30 * s_elev + 0.20 * s_slope

    if score >= 0.66:
        cls = "yüksek"
    elif score >= 0.42:
        cls = "orta"
    elif score >= 0.22:
        cls = "düşük"
    else:
        cls = "çok düşük"
    return round(score, 3), cls


def site_adjusted_risk(hazard_index: float, amp: float, amp_ref: float = 1.15) -> int:
    """hazard indeksi × büyütme (amp_ref'e normalize) → 0–100 saha-risk."""
    return int(max(0, min(100, round(hazard_index * amp / amp_ref))))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
