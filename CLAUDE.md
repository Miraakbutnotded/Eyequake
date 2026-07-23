# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dil / Language

Bu depo **Türkçe** yazılır: docstring'ler, yorumlar, script çıktıları, commit mesajları,
raporlar ve web arayüzü. Kod tanımlayıcıları (fonksiyon/değişken adları) İngilizcedir.
Yeni kod yazarken bu ayrımı koru.

## Commands

```bash
pip install -e ".[dev]"     # tek adım kurulum (pyproject; requirements.txt eşleniği)
ruff check .                 # lint (CI kapısı)
pytest -q                    # tüm testler
pytest -q --cov=eyequake --cov-report=term-missing   # CI'ın koştuğu hali
pytest tests/test_etas.py -q                          # tek dosya
pytest tests/test_etas.py::test_fit_etas_returns_valid_params -q  # tek test
python tests/test_web_honesty.py   # dürüstlük testleri standalone da koşar
```

CI (`.github/workflows/ci.yml`): Python 3.12, `ruff check .` + `pytest --cov`. Lint hatası
build'i düşürür.

## Pipeline

`scripts/` numaralı, sırayla koşan bir veri hattıdır; `src/eyequake/` ise saf,
test edilebilir kütüphane. Scriptler paketi normal import eder — çalışması için
`pip install -e .` gerekir (eski `sys.path.insert` boilerplate'i kaldırıldı).

Hattın tek kanonik girdisi `data/processed/turkey_catalog.csv`; **hemen her script
onu `data.clean.load_catalog()` ile okur** (yol/dosya adı orada tek yerde tanımlı),
yani ilk olarak bir fetch scripti koşmalıdır:

- `07_fetch_afad.py` → **kanonik** katalog (AFAD, 156k olay, temiz b≈1.0). Türkiye
  dışı IP'lerden erişilemeyebilir.
- `01_fetch_catalog.py` → USGS FDSN yedeği; aynı dosyayı yazar ama M<4'te tamlık
  artefaktı taşır (b≈0.53). Hangi kaynağın yazdığını karıştırma.

Sonra: `02_eda` (teşhis+figürler) · `03_forecast_baselines`, `05_deterministic_test`,
`08_etas`, `09_etas_spatial` (değerlendirme) · `04_hazard_map` → `06_export_web_data`
→ `10_site_layer` → `11_export_etas_params` (web verisi; 10 ve 11, 06'nın yazdığı
`web/data/hazard_cells.geojson` üzerine ekleme yapar).

`data/` ve `reports/figures/` versiyonlanmaz (regenerate edilir); `web/data/*.geojson`
ve `web/data/etas_params.json` **versiyonlanır** — web statik deploy edilir.

## Ortak katalog şeması

Tüm fetcher'lar aynı 8 kolona normalize eder — downstream'in (clean, EDA, hazard, web)
kaynaktan bağımsız çalışmasının tek sebebi bu:

```
id · time · latitude · longitude · depth · mag · magType · place
```

`data/clean.py::clean_catalog` bunun üstüne `year` ve `interevent_days` türetir, UTC'ye
çevirir, `id` üzerinden dedup eder, zamana göre sıralar. Zaman parse'ı `format="ISO8601"`
ile yapılır (AFAD bazı kayıtlarda fractional saniye taşıyor; sabit format sessiz veri
kaybına yol açar) — değiştirme. Yeni bir katalog kaynağı eklerken `_SCHEMA`'ya normalize
et; downstream'e dokunma.

## Dış servisler

Hiçbiri API anahtarı istemez, backend yok. Ağ gerektiren kod scriptlerde durur —
`src/` ve testler ağsız koşar, bu ayrımı koru.

| Servis | Nerede | Not |
|---|---|---|
| AFAD apiv2 | `data/fetch_afad.py` | Kanonik. Türkiye dışı IP'lerden geoblock olabilir |
| USGS FDSN | `data/fetch.py` | Yedek/karşılaştırma, yıllık paging |
| open-elevation | `data/elevation.py` | 500'lük POST batch; `data/raw/elevation_cache.json`'a incremental cache — rerun kaldığı yerden devam eder, cache'i silme |

## Mimari: üç track

Kod, üç bilimsel iddiayı kasten ayrı tutar (bkz. `README.md`, `reports/FINDINGS.md`):

- **Track A — hazard** (`analysis/hazard.py`, `analysis/site.py`, `web/export.py`):
  ızgara-bazlı sismisite oranı + zemin büyütmesi → göreli risk indeksi 0–100.
  Olasılıksız, savunulabilir. Ürünün çekirdeği.
- **Track B — olasılıksal oran** (`forecast/windows.py`, `baselines.py`, `etas.py`):
  ETAS/Gutenberg-Richter. Mütevazı ama gerçek beceri.
- **Track C — deterministik tahmin** (`forecast/deterministic.py`): test edildi,
  **reddedildi** (R²≈0). Kod, negatif sonucun kanıtı olarak kalır — silme.

ETAS backtest akışı (fit → pencere tahmini → climatology/persistence baseline)
`forecast/backtest.py`'de tek yerde durur; `08_etas.py` ve `09_etas_spatial.py`
yalnızca raporlama yapar. Pencere aritmetiği veya sızıntısızlık değişecekse orayı
düzenle — `tests/test_backtest.py` bunları koruyor.

Değişmez kural: her model naive baseline'a karşı `forecast/evaluate.py` ile ölçülür
(`temporal_split` — shuffle YOK, out-of-time; `poisson_deviance`; `skill_score`;
`bootstrap_mean_ci`). Yeni bir tahmin fikri eklerken baseline karşılaştırması olmadan
sonuç raporlama.

`analysis/seismology.py` (Mc, Aki b-değeri + Shi-Bolt σ) bütün track'lerin ön koşuludur:
Mc altında analiz sapmalıdır.

## Dürüstlük katmanı (bunu kırma)

Konumlandırmanın kanonik kaynağı `docs/POSITIONING.md`'dir; README/web/deck ondan
sapamaz. Dışa dönük metinde "deprem tahmini", "erken uyarı", "ne zaman/nerede/kaç
büyüklüğünde" **pozitif iddia olarak yasaktır** (disclaimer/negatif bağlamda kalır).

`tests/test_web_honesty.py` bunu statik olarak korur: disclaimer'ın varlığı, mobilde
gizlenmemesi, ETAS/R-J katsayılarının hardcode değil `web/data/etas_params.json`'dan
gelmesi, belirsizlik aralığı gösterimi, "düşük gözlem ≠ güvenli" uyarısı,
`methodology.html` içeriği. `web/index.html`, `web/app.js` veya `web/style.css`
düzenlerken bu testi koş.

Web paneli **generic Reasenberg-Jones** kullanır, projenin kendi temporal ETAS fit'ini
değil — o fit süper-kritik (branching ratio n≫1, fiziksel değil) ve yalnızca tanı
göstergesidir. `11_export_etas_params.py` her iki bloğu da provenance ile yazar.

## Bilimsel sınırlar (kodda belgeli, tekrar keşfetme)

- ETAS `_intensity_at_events` O(n²); M≥4.0 alt-kataloğu (~4.2k olay) için tasarlandı,
  tüm 156k katalog için değil.
- `analysis/site.py` Vs30'u eğimden türetir (Wald & Allen 2007) — bölgesel çözünürlük,
  mikrobölgeleme değil. Sıvılaşma bir **tarama göstergesi**; sondaj/yeraltı suyu yok.
- Rapor edilen skorlar tek 75/25 bölünmeden gelir; yeni sayı raporlarken
  `bootstrap_mean_ci` ile CI ver.

## Karar ve sonuç kayıtları

- `docs/plans/` — tasarım kararları **ve reddedilen alternatifler** gerekçesiyle
  (örn. USGS Vs30 raster'ı rasterio/GDAL ağırlığı yüzünden reddedildi). Site/BIS
  katmanında bir yaklaşım değiştirmeden önce ilgili planı oku; karar zaten
  tartışılmış olabilir.
- `reports/FINDINGS.md` — sayısal sonuçların tek doğruluk kaynağı. Bir modeli veya
  katalog kaynağını değiştiren her commit burayı da güncellemeli; aksi halde rapor
  koddan sessizce sapar (geçmişte USGS-confound'lu bölge-ETAS sayıları böyle
  düzeltildi).
- Commit mesajları Türkçe ve `tip(kapsam): açıklama` biçiminde
  (`feat`, `fix`, `docs`, `test+build`, `rigor`).

## Web

`web/` bağımlılıksız statik Leaflet MVP'sidir (build adımı yok): `index.html`,
`app.js`, `style.css` + `web/data/`. Yerelde `python -m http.server` ile aç.
