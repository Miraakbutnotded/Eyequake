"""AFAD timezone kontrat testi — bilinen M7.7 olayı ile UTC doğrulaması.

`clean.clean_catalog`, AFAD `date` alanını (fetch_afad `_normalize` bunu `time`
olarak geçirir) `pd.to_datetime(..., utc=True)` ile parse eder. AFAD `date`
alanı tz-offset taşımaz; bu yüzden saat dilimi yorumu downstream'i belirler:

  - AFAD UTC veriyorsa → saklanan zaman gerçek köken zamanına eşit, doğru.
  - AFAD yerel (Türkiye, UTC+3) veriyorsa → her zaman sessizce ~3 saat kayar,
    `recency_days` / `recent_rate` / interevent süreleri bozulur (bunlar
    risk_index ağırlıklarını besler).

Yer-gerçeği (USGS/EMSC/AFAD uzlaşısı): 6 Şubat 2023 Kahramanmaraş ana şoku
köken zamanı 2023-02-06T01:17:32–34 UTC (yerel saat 04:17 TRT, UTC+3).

Bu test bilinen bir olaya karşı kontratı kilitler: AFAD zamanları sessizce
3 saat kaymıyor, gerçekten UTC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eyequake.data.clean import clean_catalog

# 6 Şubat 2023 Kahramanmaraş (Pazarcık) M7.7 ana şoku — uzlaşılan köken zamanı.
# USGS/EMSC/AFAD ~01:17:32–34 UTC üzerinde hemfikir.
M77_ORIGIN_UTC = pd.Timestamp("2023-02-06T01:17:32Z")
M77_EVENT_ID = "543428"

# fetch_afad._normalize'in AFAD `date` alanından ortak şemaya geçirdiği HAM
# zaman dizgesi (data/raw/turkey_catalog_afad_raw.csv'deki gerçek değer):
# naive ISO, tz-offset YOK.
M77_RAW_AFAD_TIME = "2023-02-06T01:17:32"

# Köken zamanı belirsizliği (~01:17:32–34) için makul tolerans. 3 saatlik
# yerel-saat kaymasını (10800 s) yakalamak için fazlasıyla dar.
TOLERANCE_SECONDS = 120.0

_SHIFT_HINT = (
    "AFAD `date` alanı UTC değil yerel saat (Europe/Istanbul, UTC+3) sunuyor; "
    "fetch_afad._normalize bunu UTC'ye çevirmeli "
    '(tz_localize("Europe/Istanbul").tz_convert("UTC")). Bu kayma '
    "recency_days/recent_rate ve interevent zamanlamasını bozar → risk_index hatalı."
)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "turkey_catalog.csv"


def _raw_afad_m77_record() -> pd.DataFrame:
    """fetch_afad._normalize çıktısı şeklinde tek satırlık ham M7.7 kaydı.

    `time` alanı bilerek tz-offset'siz (AFAD `date` ham hali) verilir; tüm tz
    yorumu `clean_catalog` içinde olmalı.
    """
    return pd.DataFrame(
        {
            "id": [M77_EVENT_ID],
            "time": [M77_RAW_AFAD_TIME],
            "latitude": [37.27728],
            "longitude": [37.03996],
            "depth": [8.6],
            "mag": [7.7],
            "magType": ["MW"],
            "place": ["Pazarcık (Kahramanmaraş)"],
        }
    )


def test_afad_m77_fixture_parses_to_utc() -> None:
    """Ham AFAD-format M7.7 kaydı clean_catalog'dan geçince UTC köken zamanına eşit olmalı."""
    cleaned = clean_catalog(_raw_afad_m77_record())

    assert len(cleaned) == 1, "M7.7 kaydı clean_catalog sonrası düşürülmemeli."
    parsed = cleaned.loc[0, "time"]
    assert parsed.tzinfo is not None, "clean_catalog tz-aware (UTC) zaman üretmeli."

    delta_seconds = abs((parsed - M77_ORIGIN_UTC).total_seconds())
    assert delta_seconds < TOLERANCE_SECONDS, (
        f"M7.7 ayrıştırılan zaman {parsed.isoformat()}, beklenen köken "
        f"{M77_ORIGIN_UTC.isoformat()} (Δ={delta_seconds:.0f}s). "
        f"Δ≈10800s (3 saat) ise: {_SHIFT_HINT}"
    )


def test_committed_catalog_m77_is_utc() -> None:
    """Gerçek (commit'lenmiş/yerel) turkey_catalog.csv'deki M7.7 satırı ~01:17 UTC olmalı.

    Bu, fixture'dan bağımsız olarak SHIP'lenen kataloğun zaten doğru mu yoksa
    zaten bozuk mu olduğunu yakalar. Katalog .gitignore'da (data/processed/) —
    CI'da yoksa atla; fixture testi birincil kontrattır.
    """
    if not CATALOG_PATH.exists():
        pytest.skip(f"Katalog yok (gitignored): {CATALOG_PATH}")

    catalog = pd.read_csv(CATALOG_PATH)
    rows = catalog[catalog["id"].astype(str) == M77_EVENT_ID]
    if rows.empty:
        # id eşleşmesi başarısızsa zaman+büyüklükle düş.
        rows = catalog[
            (pd.to_numeric(catalog["mag"], errors="coerce") >= 7.65)
            & (catalog["time"].astype(str).str.startswith("2023-02-06"))
        ]
    assert not rows.empty, "Katalogda M7.7 (2023-02-06) ana şoku bulunamadı."

    stored = pd.to_datetime(rows["time"].iloc[0], utc=True)
    delta_seconds = abs((stored - M77_ORIGIN_UTC).total_seconds())
    assert delta_seconds < TOLERANCE_SECONDS, (
        f"Commit'lenmiş kataloğun M7.7 zamanı {stored.isoformat()}, beklenen "
        f"{M77_ORIGIN_UTC.isoformat()} (Δ={delta_seconds:.0f}s). "
        f"Δ≈10800s (3 saat) ise saklanan katalog yerel-saat-as-UTC ile bozuk: {_SHIFT_HINT}"
    )
