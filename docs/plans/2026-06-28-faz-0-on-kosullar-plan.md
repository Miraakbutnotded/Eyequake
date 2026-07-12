# Faz-0 Ön-Koşullar Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** EyeQuake'i "buildable production spec" yapan 5 bloklayıcı bilimsel-dürüstlük ön-koşulunu kapat: gerçek bir skill metriği, bilimsel `validate_outputs`, enforced integrity modülü, AFAD timezone doğrulaması, ve NOT-FOR-PRICING kısıtı.

**Architecture:** Mevcut saf-fonksiyon `src/eyequake` paketine ekleme yapılır (rewrite yok). Her ön-koşul TDD ile: önce kırmızı test, sonra minimal implementasyon. Hiçbir mevcut pipeline aşaması bozulmaz; `run_pipeline.py --check` ve `pytest` yeşil kalır. Üretilen her bilimsel iddia bir teste bağlanır.

**Tech Stack:** Python 3.11+, numpy/pandas/scipy, pytest, ruff. Mevcut modüller: `web/export.py` (risk_index), `analysis/seismology.py` (`magnitude_of_completeness`, `b_value_aki`), `forecast/evaluate.py` (`skill_score`, `temporal_split`), `forecast/baselines.py`.

**Kapsam dışı (Faz-1+):** FastAPI/PostGIS/runtime Pydantic API-envelope validator (henüz API yok → YAGNI), grid yoğunlaştırma, billing. Bunlar uçtan-uca tasarım dokümanında (`2026-06-28-end-to-end-production-system-design.md`).

**Doğrulanmış zemin (gerçek koda karşı):**
- `risk_index` = 6 sinyalin `.rank(pct=True)` ağırlıklı toplamı (`export.py:76-92`) — hiçbir baseline/skill hesabı yok.
- `scripts/06:27` `MIN_MAG=3.0`; surface + meta tutarlı 3.0. Sorun tutarsızlık değil; **Mc enforcement yokluğu** + `build_cell_index` default'unun (4.0) gerçek kullanımdan (3.0) sapması (latent).
- `src/eyequake/integrity` yok; `TRACK_C`/`BANNED_CLAIMS` sabiti yok.
- `validate_outputs` (`run_pipeline.py`) `Mc`/`b_value`/`cell_deg` çağırmıyor — string/key presence kontrolü.
- `etas_params.json:13` `skill_vs_climatology_pct: 71.1` tek hardcoded ETAS sabiti (farklı, süper-kritik model).

---

## Task 1: Gerçek spatial skill metriği (en kritik)

`risk_index` bir percentile rank; "naive baseline'ı yener" iddiasının hesaplanmış bir temeli yok. Bu task, **eğitim penceresinde** kurulan risk yüzeyinin **test penceresindeki** gelecek olayları, alana-göre-üniform (Poisson) baseline'dan daha iyi konsantre edip etmediğini ölçen bir area-skill (Molchan-tamamlayıcı / gain) metriği ekler.

**Files:**
- Create: `src/eyequake/forecast/spatial_skill.py`
- Test: `tests/test_spatial_skill.py`

**Step 1: Kırmızı test yaz** — `tests/test_spatial_skill.py`

```python
"""Spatial forecast-skill: risk yüzeyi gelecek olayları üniform-alandan iyi konsantre ediyor mu?"""

from __future__ import annotations

import numpy as np

from eyequake.forecast.spatial_skill import area_skill_score, spatial_concentration_gain


def test_perfect_concentration_scores_near_one():
    # Olaylar tamamen en yüksek riskli hücrelerde → ASS ~ +1
    counts_sorted = np.array([10, 8, 5, 0, 0, 0, 0, 0])  # risk-azalan sırada olay sayıları
    ass = area_skill_score(counts_sorted)
    assert ass > 0.7


def test_uniform_concentration_scores_near_zero():
    # Olaylar alana eşit dağılmış → skill ~ 0
    counts_sorted = np.array([3, 3, 3, 3, 3, 3, 3, 3])
    ass = area_skill_score(counts_sorted)
    assert abs(ass) < 1e-9


def test_anti_concentration_scores_negative():
    # Olaylar DÜŞÜK riskli hücrelerde (model ters) → ASS < 0
    counts_sorted = np.array([0, 0, 0, 0, 0, 5, 8, 10])
    assert area_skill_score(counts_sorted) < -0.7


def test_no_events_raises():
    import pytest
    with pytest.raises(ValueError):
        area_skill_score(np.zeros(5))


def test_gain_top_decile_beats_uniform():
    # En riskli %25 hücre, olayların >%25'ini yakalarsa gain > 1
    counts_sorted = np.array([10, 6, 2, 2, 0, 0, 0, 0])
    gain = spatial_concentration_gain(counts_sorted, area_fraction=0.25)
    assert gain > 1.0
```

**Step 2: Testi çalıştır, kırmızı doğrula**

Run: `./.venv/bin/python -m pytest tests/test_spatial_skill.py -v`
Expected: FAIL — `ModuleNotFoundError: eyequake.forecast.spatial_skill`

**Step 3: Minimal implementasyon** — `src/eyequake/forecast/spatial_skill.py`

```python
"""Spatial forecast-skill metrikleri.

Risk yüzeyinin (eğitim) gelecek olayları (test) alana-göre-üniform baseline'dan
ne kadar iyi konsantre ettiğini ölçer. Molchan/area-skill ailesi. Bu bir
TANIMLAYICI beceri ölçüsüdür — deprem tahmini iddiası değildir.
"""

from __future__ import annotations

import numpy as np


def area_skill_score(counts_sorted_by_risk: np.ndarray) -> float:
    """Area Skill Score (ASS) ∈ [-1, 1]. 0 = üniform-alan baseline kadar.

    Hücreler azalan risk_index'e göre sıralı; ``counts_sorted_by_risk`` her
    hücredeki gelecek-olay sayısı. Eşit-alan hücre varsayımı (kümülatif hücre
    oranı = kümülatif alan oranı). ASS = 2*AUC - 1, AUC = yakalanan-olay
    eğrisinin altında kalan alan (kümülatif alan oranına karşı).
    """
    counts = np.asarray(counts_sorted_by_risk, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("En az bir gelecek olay gerekir (toplam=0).")
    n = counts.size
    cum_events = np.concatenate(([0.0], np.cumsum(counts) / total))
    area_frac = np.linspace(0.0, 1.0, n + 1)
    auc = float(np.trapz(cum_events, area_frac))
    return round(2.0 * auc - 1.0, 4)


def spatial_concentration_gain(counts_sorted_by_risk: np.ndarray, area_fraction: float = 0.25) -> float:
    """En riskli ``area_fraction`` hücrenin yakaladığı olay oranı / area_fraction.

    Gain > 1 → o alan diliminde üniform-alandan iyi. (Lift / probability gain.)
    """
    counts = np.asarray(counts_sorted_by_risk, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("En az bir gelecek olay gerekir (toplam=0).")
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError("area_fraction (0, 1] aralığında olmalı.")
    k = max(1, int(round(area_fraction * counts.size)))
    captured = counts[:k].sum() / total
    return round(captured / area_fraction, 4)
```

> Not: numpy 2.0'da `np.trapz` → `np.trapezoid`. Hata alırsan importu `trapz = getattr(np, "trapezoid", np.trapz)` ile koru.

**Step 4: Testi çalıştır, yeşil doğrula**

Run: `./.venv/bin/python -m pytest tests/test_spatial_skill.py -v`
Expected: PASS (5/5)

**Step 5: Katalog-seviyesi out-of-time skill (entegrasyon)**

`tests/test_spatial_skill.py`'ye ekle — sentetik katalogla train/test ayrımı + `build_cell_index` üzerinden gerçek akış:

```python
def test_out_of_time_skill_on_clustered_synthetic_catalog():
    import pandas as pd
    from eyequake.config import TURKEY
    from eyequake.web.export import build_cell_index
    from eyequake.forecast.spatial_skill import evaluate_surface_skill

    rng = np.random.default_rng(0)
    # İki kümeli sentetik katalog: train + test aynı bölgelerde yoğun
    n = 600
    hot = rng.choice([0, 1], size=n)
    lat = np.where(hot == 1, 39.0, 37.0) + rng.normal(0, 0.1, n)
    lon = np.where(hot == 1, 28.0, 36.0) + rng.normal(0, 0.1, n)
    t = pd.date_range("2000-01-01", "2020-01-01", periods=n, tz="UTC")
    df = pd.DataFrame({
        "time": t, "latitude": lat, "longitude": lon,
        "mag": rng.uniform(4.0, 6.0, n), "depth": rng.uniform(5, 20, n),
    })
    res = evaluate_surface_skill(df, TURKEY, min_mag=4.0, test_frac=0.25)
    # Sismisite kalıcıysa risk yüzeyi gelecek olayları üniformdan iyi konsantre etmeli
    assert res["area_skill_score"] > 0.2
    assert res["n_test_events"] > 0
```

Implementasyona ekle (`spatial_skill.py`):

```python
def evaluate_surface_skill(df, region, *, min_mag: float = 4.0, cell_deg: float = 0.25, test_frac: float = 0.25) -> dict:
    """Out-of-time spatial skill: train penceresinde risk_index kur, test
    penceresindeki olayları o sıralamaya göre say, ASS + gain döndür."""
    import pandas as pd
    from ..web.export import build_cell_index, _to_utc

    d = df.copy()
    d["time"] = _to_utc(d["time"])
    d = d.sort_values("time")
    cut = d["time"].quantile(1.0 - test_frac)
    train, test = d[d["time"] <= cut], d[d["time"] > cut]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("train/test penceresi boş.")

    cells = build_cell_index(train, region, cell_deg=cell_deg, min_mag=min_mag)
    # Test olaylarını train hücre binlerine ata
    tt = test[test["mag"] >= min_mag].copy()
    tt["lat_bin"] = np.floor((tt["latitude"] - region.min_lat) / cell_deg).astype(int)
    tt["lon_bin"] = np.floor((tt["longitude"] - region.min_lon) / cell_deg).astype(int)
    key = list(zip(tt["lat_bin"], tt["lon_bin"]))
    from collections import Counter
    cnt = Counter(key)
    counts_sorted = np.array([
        cnt.get((int(r["lat_bin"]), int(r["lon_bin"])), 0)
        for _, r in cells.iterrows()
    ], dtype=float)
    return {
        "area_skill_score": area_skill_score(counts_sorted) if counts_sorted.sum() > 0 else None,
        "gain_top25": spatial_concentration_gain(counts_sorted, 0.25) if counts_sorted.sum() > 0 else None,
        "n_train_cells": int(len(cells)),
        "n_test_events": int(counts_sorted.sum()),
    }
```

Run: `./.venv/bin/python -m pytest tests/test_spatial_skill.py -v` → PASS (6/6)

**Step 6: Ruff + commit**

```bash
./.venv/bin/ruff check src/eyequake/forecast/spatial_skill.py tests/test_spatial_skill.py
git add src/eyequake/forecast/spatial_skill.py tests/test_spatial_skill.py
git commit -m "rigor: spatial forecast-skill metriği (area-skill + gain, out-of-time)"
```

---

## Task 2: Bilimsel `validate_outputs` + Mc enforcement

`validate_outputs` bugün sadece string/key presence kontrol ediyor. Bu task ona **sayısal/bilimsel tutarlılık** ekler: `min_mag ≥ Mc`, meta↔surface `min_mag`/`cell_deg` tutarlılığı, plausible `b_value`. Ayrıca `build_cell_index`'in default-min_mag sapmasını kapatır.

**Files:**
- Modify: `src/eyequake/web/export.py:25-28` (min_mag default'u kaldır → zorunlu kıl)
- Modify: `scripts/run_pipeline.py` (`validate_outputs` + yeni `validate_science`)
- Modify: `scripts/06_export_web_data.py` (Mc'yi meta'ya stamp et)
- Test: `tests/test_run_pipeline.py`

**Step 1: Kırmızı test** — `tests/test_run_pipeline.py`'ye ekle

```python
def test_min_mag_must_be_at_or_above_mc():
    from scripts.run_pipeline import assert_min_mag_complete
    import pytest
    # Mc=2.7 iken min_mag=3.0 geçerli
    assert_min_mag_complete(min_mag=3.0, mc=2.7)
    # min_mag Mc'nin altındaysa → reporting bias, reddet
    with pytest.raises(Exception):
        assert_min_mag_complete(min_mag=2.0, mc=2.7)


def test_build_cell_index_requires_explicit_min_mag():
    import inspect
    from eyequake.web.export import build_cell_index
    sig = inspect.signature(build_cell_index)
    assert sig.parameters["min_mag"].default is inspect._empty, \
        "min_mag explicit verilmeli (latent default-divergence guard)"


def test_meta_cell_deg_matches_hazard_polygons(tmp_path):
    from scripts.run_pipeline import assert_cell_deg_consistent
    # meta cell_deg ile poligon kenarı tutmazsa hata
    import pytest
    with pytest.raises(Exception):
        assert_cell_deg_consistent(meta_cell_deg=0.25, polygon_edge_deg=0.50)
    assert_cell_deg_consistent(meta_cell_deg=0.25, polygon_edge_deg=0.25)
```

**Step 2: Kırmızı doğrula**

Run: `./.venv/bin/python -m pytest tests/test_run_pipeline.py -k "min_mag or cell_deg" -v`
Expected: FAIL — `ImportError: cannot import name 'assert_min_mag_complete'` + `build_cell_index` default hâlâ 4.0.

**Step 3: `build_cell_index` min_mag zorunlu** — `export.py:25-28`

```python
def build_cell_index(
    df: pd.DataFrame, region: Region, *, cell_deg: float = 0.25,
    min_mag: float, recent_years: int = 10,
) -> pd.DataFrame:
```

> `*` ile `min_mag` keyword-only + default'suz. Çağıranlar: `scripts/06_export_web_data.py:34` zaten `min_mag=MIN_MAG` veriyor (kw) → uyumlu. `tests/test_web_export.py` ve `tests/test_spatial_skill.py` çağrılarını `min_mag=...` kw'ye çevir (gerekirse).

**Step 4: `run_pipeline.py`'ye bilim-assert'leri ekle**

```python
def assert_min_mag_complete(min_mag: float, mc: float) -> None:
    if min_mag < mc:
        raise PipelineError(
            f"min_mag ({min_mag}) Mc ({mc}) altında — eksik katalog, raporlama-bias riski."
        )


def assert_cell_deg_consistent(meta_cell_deg: float, polygon_edge_deg: float, tol: float = 1e-6) -> None:
    if abs(meta_cell_deg - polygon_edge_deg) > tol:
        raise PipelineError(
            f"meta cell_deg ({meta_cell_deg}) hazard poligon kenarı ({polygon_edge_deg}) ile uyumsuz."
        )


def polygon_edge_deg(feature: dict) -> float:
    ring = feature["geometry"]["coordinates"][0]
    xs = [pt[0] for pt in ring]
    return round(max(xs) - min(xs), 6)


def validate_science(root: Path = ROOT) -> dict:
    """Sayısal/bilimsel tutarlılık: min_mag≥Mc, cell_deg, b_value band."""
    import numpy as np
    import pandas as pd
    from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness

    catalog = pd.read_csv(root / PROCESSED_DIR / "turkey_catalog.csv")
    meta = load_json(root / WEB_DATA_DIR / "meta.json")
    mags = catalog["mag"].to_numpy(dtype=float)
    mc = magnitude_of_completeness(mags)

    assert_min_mag_complete(float(meta["min_mag"]), mc)
    feat = first_feature(root / WEB_DATA_DIR / "hazard_cells.geojson")
    assert_cell_deg_consistent(float(meta["cell_deg"]), polygon_edge_deg(feat))

    bv = b_value_aki(mags, mc).b
    if not (0.6 <= bv <= 1.4):
        raise PipelineError(f"b-value ({bv}) plausible bant [0.6, 1.4] dışında — katalog/Mc sorunu.")
    return {"mc": mc, "b_value": bv, "min_mag": meta["min_mag"], "cell_deg": meta["cell_deg"]}
```

`main()` içinde, `validate_outputs` çağrısından sonra (export/check kolunda):

```python
        if options.export_web or options.check:
            summary = validate_outputs(ROOT, require_site_layer=not options.skip_site_layer)
            science = validate_science(ROOT)          # YENİ — bilim kapısı
            summary.update(science)
            print_summary(summary, options)
```

**Step 5: Mc'yi meta'ya stamp et** — `scripts/06_export_web_data.py` (`meta` dict'ine)

```python
    from eyequake.analysis.seismology import magnitude_of_completeness
    mc = magnitude_of_completeness(df["mag"].to_numpy(dtype=float))
    meta = {
        ...
        "min_mag": MIN_MAG,
        "mc": mc,                # YENİ — Mc provenance
        ...
    }
```

`run_pipeline.validate_outputs`'taki `require_keys("meta", ...)` setine `"mc"` ekle.

**Step 6: Yeşil + tam suite**

Run: `./.venv/bin/python -m pytest tests/test_run_pipeline.py -v` → PASS
Run: `./.venv/bin/python scripts/run_pipeline.py --check` → bilim kapısı dahil PASS (Mc + b_value yazdırır)

> Eğer `--check` `min_mag (3.0) Mc altında` ile patlarsa, bu **gerçek bir bulgu** — MIN_MAG'ı Mc'ye çıkar veya kararı dokümante et. Sessizce gevşetme.

**Step 7: Ruff + commit**

```bash
./.venv/bin/ruff check .
git add src/eyequake/web/export.py scripts/run_pipeline.py scripts/06_export_web_data.py tests/test_run_pipeline.py
git commit -m "rigor: validate_outputs bilimsel tutarlılık kapısı (min_mag≥Mc, cell_deg, b-value) + Mc provenance"
```

---

## Task 3: Enforced integrity modülü (negation-aware banned-claim)

Track-C "rejected" ve yasaklı-iddia listesi şu an yalnız prose. Bu task bunları **tek-doğruluk-kaynağı sabitlerine** + **negation-aware lintere** dönüştürür ve `test_web_honesty.py`'nin 8 hardcoded substring assertion'ını bu lintere bağlar. TR casefold (İ/ı) tuzağı ele alınır.

**Files:**
- Create: `src/eyequake/integrity/__init__.py`
- Create: `src/eyequake/integrity/claims.py`
- Test: `tests/test_integrity.py`
- Modify: `tests/test_web_honesty.py` (linteri kullan)

**Step 1: Kırmızı test** — `tests/test_integrity.py`

```python
"""Integrity sabitleri + negation-aware banned-claim linter."""

from __future__ import annotations

from eyequake.integrity.claims import BANNED_CLAIMS, TRACK_C_STATUS, scan_text


def test_track_c_is_rejected_constant():
    assert TRACK_C_STATUS == "rejected"


def test_positive_banned_claim_is_flagged():
    hits = scan_text("Sistemimiz deprem tahmini yapar ve ne zaman olacağını söyler.")
    assert any("deprem tahmini" in h.phrase for h in hits)


def test_negated_disclaimer_is_allowed():
    # POSITIONING zorunlu disclaimer'ı: yasaklı ifade NEGATİF bağlamda → flag YOK
    hits = scan_text("Bu bir deprem tahmini DEĞİLDİR; göreli risk indeksidir.")
    assert hits == []


def test_turkish_casefold_uppercase_claim_flagged():
    hits = scan_text("DEPREM TAHMİNİ MOTORU")
    assert len(hits) >= 1


def test_neural_net_prediction_flagged():
    assert scan_text("yapay sinir ağı ile deprem tahmini") != []


def test_clean_text_no_hits():
    assert scan_text("Göreli sismik risk indeksi (0–100), tanımlayıcı bir göstergedir.") == []
```

**Step 2: Kırmızı doğrula**

Run: `./.venv/bin/python -m pytest tests/test_integrity.py -v`
Expected: FAIL — `ModuleNotFoundError: eyequake.integrity`

**Step 3: Implementasyon** — `src/eyequake/integrity/claims.py`

```python
"""Bilimsel-dürüstlük değişmezleri — tek doğruluk kaynağı.

POSITIONING.md yasaklı ifadeleri + Track-C statüsü burada SABİT olarak yaşar;
prose değil, kodla enforce edilir. ``scan_text`` negation-aware: yasaklı ifade
yalnızca POZİTİF iddia olarak geçtiğinde bayraklanır; negatif/disclaimer bağlamı
('... DEĞİLDİR') serbesttir (POSITIONING'in istisnası).
"""

from __future__ import annotations

from dataclasses import dataclass

# Track C (deterministik tahmin) reports/FINDINGS.md'de test edilip reddedildi.
TRACK_C_STATUS = "rejected"

# Dışa dönük metinde POZİTİF iddia olarak yasak (POSITIONING.md "Yasaklı kelimeler").
BANNED_CLAIMS = (
    "deprem tahmini",
    "deprem öngörüsü",
    "erken uyarı",
    "ne zaman deprem",
    "nerede deprem",
)

# Negation pencere işaretleri (yasaklı ifadeden SONRA gelirse → disclaimer, serbest).
_NEGATIONS = ("değil", "degil", "yapmaz", "mümkün değil", "mumkun degil")
_WINDOW = 40  # karakter


def _fold(s: str) -> str:
    # Türkçe-güvenli küçültme: İ/I ayrımı casefold ile korunur.
    return s.replace("İ", "i").replace("I", "ı").casefold()


@dataclass(frozen=True)
class ClaimHit:
    phrase: str
    index: int


def scan_text(text: str) -> list[ClaimHit]:
    """Pozitif yasaklı-iddia geçişlerini döndürür (negatif bağlam hariç)."""
    folded = _fold(text)
    hits: list[ClaimHit] = []
    for phrase in BANNED_CLAIMS:
        p = _fold(phrase)
        start = 0
        while (i := folded.find(p, start)) != -1:
            window = folded[i + len(p): i + len(p) + _WINDOW]
            if not any(neg in window for neg in _NEGATIONS):
                hits.append(ClaimHit(phrase=phrase, index=i))
            start = i + len(p)
    return hits
```

`src/eyequake/integrity/__init__.py`:

```python
from .claims import BANNED_CLAIMS, TRACK_C_STATUS, ClaimHit, scan_text

__all__ = ["BANNED_CLAIMS", "TRACK_C_STATUS", "ClaimHit", "scan_text"]
```

**Step 4: Yeşil doğrula**

Run: `./.venv/bin/python -m pytest tests/test_integrity.py -v` → PASS (6/6)

**Step 5: `test_web_honesty.py`'yi lintere bağla**

`tests/test_web_honesty.py`'ye ekle (mevcut 8 testi silmeden — yeni katman):

```python
from eyequake.integrity import scan_text

def test_outward_web_copy_has_no_positive_banned_claim():
    for name in ("index.html", "methodology.html"):
        hits = scan_text(_read(name))
        assert hits == [], f"{name} pozitif yasaklı-iddia içeriyor: {[h.phrase for h in hits]}"
```

> Bu, mevcut disclaimer'ların ('deprem tahmini DEĞİLDİR') negation-aware geçtiğini de doğrular — pozitif-corpus + negatif-corpus dengesi (kritiğin istediği).

Run: `./.venv/bin/python -m pytest tests/test_web_honesty.py tests/test_integrity.py -v` → PASS

**Step 6: Ruff + commit**

```bash
./.venv/bin/ruff check src/eyequake/integrity tests/test_integrity.py tests/test_web_honesty.py
git add src/eyequake/integrity tests/test_integrity.py tests/test_web_honesty.py
git commit -m "rigor: enforced integrity modülü (negation-aware banned-claim, Track-C sabiti, TR casefold)"
```

---

## Task 4: AFAD timezone contract testi

`clean_catalog` ISO8601 UTC varsayar. AFAD `date` alanı yerel (UTC+3) ise tüm `recency_days`/`recent_rate` sinyalleri ~3 saat kayar. Bu task, bilinen bir olaya (6 Şubat 2023 Kahramanmaraş M7.7, origin **2023-02-06T01:17:34Z**) karşı pinli bir fixture ile parse'ı doğrular.

**Files:**
- Test: `tests/test_afad_timezone.py`
- (Gerekirse) Modify: `src/eyequake/data/fetch_afad.py` (timezone normalizasyonu)

**Step 1: Gerçek AFAD ham kaydını yakala** (manuel, bir kerelik)

```bash
./.venv/bin/python scripts/07_fetch_afad.py   # veya watch_afad ile dar pencere
# data/processed/turkey_catalog.csv içinde M7.7 Kahramanmaraş satırını bul:
grep -i "2023-02-06" data/processed/turkey_catalog.csv | grep "7.7" | head
```

Bulunan `time` değerini Step 2'deki `RAW_TIME` sabitine yapıştır.

**Step 2: Kırmızı/doğrulama testi** — `tests/test_afad_timezone.py`

```python
"""AFAD timezone contract: bilinen olay UTC'ye doğru parse olmalı."""

from __future__ import annotations

import pandas as pd

from eyequake.data.clean import clean_catalog

# 6 Şubat 2023 Kahramanmaraş M7.7 — kanonik origin: 2023-02-06T01:17:34Z (USGS/EMSC).
RAW_TIME = "2023-02-06T01:17:34.000Z"   # ← Step 1'de bulunan GERÇEK AFAD değeriyle değiştir
EXPECTED_UTC = pd.Timestamp("2023-02-06T01:17:34Z")


def test_known_mainshock_parses_to_expected_utc():
    raw = pd.DataFrame([{
        "id": "afad_kmaras_2023", "time": RAW_TIME,
        "latitude": 37.288, "longitude": 37.043, "depth": 8.6,
        "mag": 7.7, "magType": "Mw", "place": "Kahramanmaraş",
    }])
    out = clean_catalog(raw)
    delta = abs((out.loc[0, "time"] - EXPECTED_UTC).total_seconds())
    assert delta < 120, (
        f"AFAD time UTC değil olabilir: parse={out.loc[0,'time']} beklenen={EXPECTED_UTC} "
        f"(Δ={delta/3600:.2f}h). 3h sapma → AFAD yerel saat veriyor, fetch_afad'da +03 düzelt."
    )
```

**Step 3: Çalıştır + karar**

Run: `./.venv/bin/python -m pytest tests/test_afad_timezone.py -v`
- PASS → AFAD UTC veriyor, contract sabitlendi. Bitti.
- FAIL Δ≈3h → AFAD yerel saat veriyor. `fetch_afad.py` normalizasyonuna `tz_localize("Europe/Istanbul").tz_convert("UTC")` ekle, testi yeşile çevir, **recency sinyallerinin etkilendiğini** commit mesajında belirt.

**Step 4: Ruff + commit**

```bash
./.venv/bin/ruff check tests/test_afad_timezone.py
git add tests/test_afad_timezone.py
git commit -m "rigor: AFAD timezone contract testi (bilinen M7.7 → UTC doğrulaması)"
```

---

## Task 5: NOT-FOR-PRICING kısıtı (E&O guard)

Ürün "göreli risk indeksi" — absolute loss'a kalibre değil. Bu task, premium-rating'e doğrudan girdi olarak kullanımını yasaklayan enforced bir kullanım-kısıtı bayrağı ekler (sözleşme + UI/meta).

**Files:**
- Modify: `scripts/06_export_web_data.py` (meta'ya `use_restriction`)
- Modify: `scripts/run_pipeline.py` (`validate_outputs` → kısıtı zorla)
- Test: `tests/test_run_pipeline.py`

**Step 1: Kırmızı test** — `tests/test_run_pipeline.py`'ye

```python
def test_meta_carries_not_for_pricing_restriction(tmp_path):
    from scripts.run_pipeline import assert_use_restriction
    import pytest
    assert_use_restriction({"use_restriction": "göreli triage / accumulation control — premium rating için DEĞİL"})
    with pytest.raises(Exception):
        assert_use_restriction({})  # kısıt yoksa reddet
    with pytest.raises(Exception):
        assert_use_restriction({"use_restriction": "fiyatlama için hazır"})  # yanlış içerik
```

**Step 2: Kırmızı doğrula**

Run: `./.venv/bin/python -m pytest tests/test_run_pipeline.py -k use_restriction -v` → FAIL (ImportError)

**Step 3: Implementasyon**

`scripts/06_export_web_data.py` meta dict'ine:

```python
        "use_restriction": (
            "Göreli triage / portföy sıralama / accumulation control. "
            "Absolute loss'a kalibre DEĞİL — premium rating için doğrudan girdi DEĞİLDİR."
        ),
```

`scripts/run_pipeline.py`:

```python
def assert_use_restriction(meta: dict) -> None:
    val = str(meta.get("use_restriction", "")).lower()
    if "rating" not in val and "değil" not in val:
        raise PipelineError("meta.use_restriction NOT-FOR-PRICING kısıtını taşımalı (E&O guard).")
```

`validate_outputs` içinde `require_keys("meta", meta, {...})` setine `"use_restriction"` ekle ve hemen ardından `assert_use_restriction(meta)` çağır.

**Step 4: Yeşil + check**

Run: `./.venv/bin/python -m pytest tests/test_run_pipeline.py -v` → PASS
Run: `./.venv/bin/python scripts/run_pipeline.py --check` → PASS (kısıt dahil)

**Step 5: Ruff + commit**

```bash
./.venv/bin/ruff check .
git add scripts/06_export_web_data.py scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "rigor: NOT-FOR-PRICING kullanım kısıtı (E&O guard) + meta enforcement"
```

---

## Kapanış: tam yeniden-üretim + kapı doğrulaması

Tüm task'lar bitince:

```bash
./.venv/bin/ruff check .
./.venv/bin/python -m pytest --cov=eyequake --cov-report=term-missing
./.venv/bin/python scripts/run_pipeline.py --skip-fetch --skip-site-layer --export-web
```

Expected: ruff temiz, tüm testler yeşil (yeni: spatial_skill 6, integrity 6, run_pipeline +4, afad_timezone 1, web_honesty +1), `--export-web` bilim kapısı (Mc/b-value/min_mag/cell_deg/use_restriction) dahil PASS.

**Tamamlanma kanıtı:** Faz-0'ın 5 bloklayıcısı artık koda bağlı: gerçek skill metriği (Task 1), bilimsel validate (Task 2), enforced integrity (Task 3), timezone contract (Task 4), pricing guard (Task 5). Bu noktada uçtan-uca tasarım dokümanının Faz-1'i (FastAPI/PostGIS pilot) buildable hale gelir.

---

## Notlar
- Tasarım dokümanı `2026-06-28-end-to-end-production-system-design.md` §0, kritiğin min_mag iddiasını "self-inconsistent surface" olarak aktarmıştı; doğrulama gösterdi ki üretim tutarlı (06 explicit MIN_MAG=3.0) — gerçek açık **Mc-enforcement yokluğu**ydu (Task 2 onu kapatır). §0 madde 2 bu yönde güncellenebilir.
- Runtime Pydantic honesty-envelope validator (API yanıt zarfı) kasıtla Faz-1'e bırakıldı — henüz API yok (YAGNI). Integrity sabitleri (Task 3) o validator'ın temelini oluşturur.
