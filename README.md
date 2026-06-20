# EyeQuake — Sismik Risk & Bina Kırılganlık İstihbaratı (araştırma prototipi)

> 6 Şubat 2023 depremleri sonrası doğan EyeQuake projesinin **araştırma prototipi**.
> Amaç: *"Geçmiş deprem kataloğundan deprem gerçekten öngörülebilir mi?"* sorusunu
> dürüst baseline'lara karşı, sıkı zaman-serisi doğrulamasıyla test etmek.
>
> **Ürün konumu** (bkz. [`docs/POSITIONING.md`](docs/POSITIONING.md)): parsel-bazlı
> sismik risk + bina-kırılganlık istihbaratı. Beachhead = **sigorta/reasürans**
> (parsel-risk → fiyatlama). Hazard = commodity, **BIS/bina-kırılganlık = çekirdek
> farklılaştırıcı**. Bu bir **deprem tahmini değil** — Track C bunu test edip reddetti (↓).

## Bilimsel duruş (önemli)

Deprem **tahmini** üç farklı bilimsel zeminde durur. Bu prototip üçünü de yan yana,
aynı veriyle ve aynı dürüst değerlendirmeyle kurar:

| Track | Soru | Bilimsel statü |
|-------|------|----------------|
| **A — Tehlike haritası** | "Bu bölge ne kadar riskli?" (uzun vade, olasılıksız) | ✅ Standart (PSHA temeli) |
| **B — Olasılıksal öngörü** | "Önümüzdeki pencerede beklenen deprem oranı/büyüklüğü?" | ✅ Bilimsel (ETAS, Gutenberg-Richter) |
| **C — Deterministik tahmin** | "Şu tarihte, şu yerde, şu büyüklükte deprem" | ⚠️ Bilimsel konsensüs: kısa-vadeli deterministik tahmin **çözülmemiş** |

**Track C'yi reddetmiyoruz — test ediyoruz.** Her track, naive baseline'lara karşı
ölçülür (Poisson background rate, "ortalamayı tahmin et", persistence). Bir model
yalnızca bu baseline'ları **istatistiksel olarak yenebiliyorsa** anlamlıdır. Veri
konuşur, pazarlama değil.

## Veri

- **Birincil (kanonik):** AFAD ulusal ağ kataloğu — Türkiye, 1990–bugün, 156k+ olay.
  Temiz Gutenberg-Richter b≈1.0 (sağlam katalog), il/ilçe metadata. `scripts/07_fetch_afad.py`.
- **Karşılaştırma/yedek:** USGS FDSN Event API (açık, anahtarsız). `scripts/01_fetch_catalog.py`.
  USGS Türkiye'de M<4 için zaman-değişken tamlık taşır (b≈0.53 artefakt) — bkz. `reports/FINDINGS.md`.
- Bina-seviyesi veri (BIS) ileri aşama — şirket raporundaki bilinen veri açığı.

## Yapı

```
src/eyequake/
  config.py            # bölge bbox, sabitler, yollar
  data/fetch.py        # USGS FDSN çekici (yıllık paging)
  data/fetch_afad.py   # AFAD ulusal katalog çekici (ortak şemaya normalize)
  data/clean.py        # normalizasyon, dedup, türetilmiş alanlar
  analysis/seismology.py  # Mc, b-değeri, FMD (Gutenberg-Richter)
  analysis/hazard.py      # Track A — uzaysal sismisite oran alanı
  forecast/windows.py · baselines.py · evaluate.py · deterministic.py  # Track B/C
  web/export.py        # ön yüz için GeoJSON üretimi
scripts/
  01_fetch_catalog.py  # USGS indir   ·  07_fetch_afad.py     # AFAD indir (kanonik)
  02_eda.py            # EDA + teşhis  ·  04_hazard_map.py     # Track A harita
  03_forecast_baselines.py · 05_deterministic_test.py         # Track B/C eval
  06_export_web_data.py    # web GeoJSON
web/                     # statik risk haritası MVP (Leaflet) — deploy-hazır
data/{raw,processed}/    # üretilir, versiyonlanmaz
reports/                 # FINDINGS.md + figures/
```

## Kurulum

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Kullanım

```bash
./.venv/bin/python scripts/01_fetch_catalog.py
```

## Ekip

Bora Esen (ODTÜ İstatistik) · Alp Özdemir (TED Bilgisayar Müh.)
