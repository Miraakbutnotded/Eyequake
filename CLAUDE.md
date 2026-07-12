# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ne olduğu

EyeQuake = Python araştırma prototipi + statik web MVP (Leaflet). Çekirdek soru:
*"Geçmiş deprem kataloğundan deprem gerçekten öngörülebilir mi?"* — dürüst
baseline'lara karşı, sıkı zaman-serisi doğrulamasıyla. **Bu bir deprem tahmini
ürünü değil.** Ürün konumu: parsel-bazlı sismik risk + bina-kırılganlık istihbaratı
(beachhead: sigorta/reasürans). Detay: `README.md`, `docs/POSITIONING.md`.

## Komutlar

```bash
# Kurulum (editable + dev araçları: pytest, pytest-cov, ruff)
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"

# Test (pyproject addopts="-q", testpaths=tests/)
./.venv/bin/python -m pytest
./.venv/bin/python -m pytest --cov=eyequake --cov-report=term-missing   # CI ile aynı

# Tek test / tek dosya
./.venv/bin/python -m pytest tests/test_etas.py
./.venv/bin/python -m pytest tests/test_etas.py::test_<behavior> -v

# Lint (CI gate — line-length 100, target py311)
./.venv/bin/ruff check .

# Tam yeniden-üretim hattı (fetch → web export → site layer → ETAS → ruff → pytest)
./.venv/bin/python scripts/run_pipeline.py --source afad --export-web

# Sadece mevcut çıktıların kontrat doğrulaması (fetch/build yok)
./.venv/bin/python scripts/run_pipeline.py --check

# Yükseklik API'si / cache sorunluysa site katmanını atla
./.venv/bin/python scripts/run_pipeline.py --skip-fetch --skip-site-layer --export-web

# Yakın-canlı AFAD feed kontrolü (volatil web/data/live_*.* yazar)
./.venv/bin/python scripts/watch_afad.py --once --lookback-minutes 120 --min-mag 0
```

CI (`.github/workflows`): Python 3.12, `pip install -e ".[dev]"` → `ruff check .` → `pytest --cov`. Coverage eşiği yok ama eksik satırlar yazdırılır.

## Mimari — bilmek için birden çok dosya okumak gerekenler

### Üç Track doktrini (projenin omurgası)
Tek katalog + tek dürüst değerlendirme, üç bilimsel zeminde yan yana kurulur:
- **Track A — Tehlike haritası** (`analysis/hazard.py`): uzaysal sismisite oran alanı. Olasılıksız, uzun vade. ✅ standart (PSHA temeli).
- **Track B — Olasılıksal öngörü** (`forecast/etas.py`, `windows.py`, `baselines.py`): ETAS + Gutenberg-Richter. Pencere-bazlı beklenen oran/büyüklük. ✅ bilimsel.
- **Track C — Deterministik tahmin** (`forecast/deterministic.py`, `evaluate.py`): "şu tarih/yer/büyüklük". ⚠️ test edilip **reddedildi** — bulgular `reports/FINDINGS.md`.

**Değişmez kural:** her model naive baseline'a karşı ölçülür (Poisson background, "ortalamayı tahmin et", persistence). Bir model ancak baseline'ı **istatistiksel olarak yenebiliyorsa** anlamlı sayılır. Forecast/hazard çıktısının yanında belirsizlik, provenance ve baseline karşılaştırması açıkça durmalı.

### Veri akışı
`scripts/07_fetch_afad.py` (AFAD, **kanonik**) → `data/clean.py:clean_catalog` (UTC parse, dedup-by-id, türetilmiş `year`/`interevent_days`) → `analysis/` + `forecast/` → `web/export.py` (GeoJSON: `build_cell_index` → `cells_to_geojson`, `events_to_geojson`). `data/{raw,processed}/` ve `reports/figures/` üretilir, **versiyonlanmaz** (`.gitignore`).

- **AFAD = kanonik** Türkiye katalogu (temiz Gutenberg-Richter b≈1.0). USGS (`scripts/01`, `data/fetch.py`) yalnızca karşılaştırma/yedek — Türkiye M<4'te tamlık artefaktı taşır (b≈0.53). Yeni iş AFAD'ı varsayar.
- Bölge/fay tanımları ve katalog varsayılanları (`CATALOG_START_YEAR=1990`, `DEFAULT_MIN_MAGNITUDE=2.5`) tek yerde: `src/eyequake/config.py`.

### Pipeline orkestratörü + çıktı kontratı (`scripts/run_pipeline.py`)
Numaralı scriptleri (`06_export_web_data`, `10_site_layer`, `11_export_etas_params`, fetch) sıralı koşturur, sonra `ruff`+`pytest` kalite kapılarını uygular. `validate_outputs` web çıktılarının **şeklini ve dürüstlüğünü** zorlar:
- `web/data/meta.json` ve `summary.json`: `source` == `"AFAD"`.
- `meta.json` ve `etas_params.json` `disclaimer`'ı **"tahmini" kelimesini içermeli** (tahmin sınırını belirtir).
- `hazard_cells.geojson` her feature'da zorunlu prop seti (`risk_model`, `risk_index`, `annual_rate`, … ; site layer'la `site_risk`, `site_class`, `amp`, `liq_class`).

Web `web/data/` çıktısını değiştiriyorsan bu kontratı kırmadığını `--check` ile doğrula.

### Bilimsel dürüstlük testleri (kırma)
`tests/test_web_honesty.py` UI metnini (`web/index.html`, `methodology.html`) denetler — disclaimer/wording bilimsel dürüstlüğü ve güvenliği korur. Frontend metni "deterministik deprem tahmini" ima edemez. Bu testler ve `validate_outputs` kontratı, modeller iyimser pazarlamaya kaymasın diye var.

## Yaparken dikkat
- Numaralı `scripts/NN_*.py` sıralı hat aşamaları — sıralamayı koru, src/eyequake'i kütüphane olarak çağırır.
- Test dosyası `tests/test_<area>.py`, test `test_<behavior>`. Deterministik matematik için küçük sentetik diziler; forecast/hazard/cleaning değişiminde boş-geçmiş, bozuk-büyüklük, baseline-karşılaştırma edge case'lerini kapsa.
- Commit prefiksleri: `feat:`, `fix:`, `docs:`, `test+build:`, `rigor:`. Konu tek satır, bilimsel/ürün etkisini anlat. Veri-kaynağı veya model-iddiası değişimini açıkça belirt.
- Sır, `.venv/`, ham/işlenmiş katalog, üretilen figür **commit etme**.
