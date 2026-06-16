# EyeQuake — Sismik Risk & Deprem Öngörü Araştırma Prototipi

> 6 Şubat 2023 depremleri sonrası doğan EyeQuake projesinin **araştırma prototipi**.
> Amaç: *"Geçmiş deprem kataloğundan deprem gerçekten öngörülebilir mi?"* sorusunu
> dürüst baseline'lara karşı, sıkı zaman-serisi doğrulamasıyla test etmek.

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

- **Birincil:** USGS FDSN Event API (açık, anahtarsız) — Türkiye bounding box, 1990–bugün.
- **Otoriter alternatif (sonra):** AFAD / Kandilli (KOERI) ulusal katalogları.
- Bina-seviyesi veri (BIS) ileri aşama — şirket raporundaki bilinen veri açığı.

## Yapı

```
src/eyequake/
  config.py          # bölge bbox, sabitler, yollar
  data/
    fetch.py         # USGS FDSN katalog çekici (yıllık paging)
    clean.py         # normalizasyon, dedup, türetilmiş alanlar
scripts/
  01_fetch_catalog.py   # ham katalogu indir
data/{raw,processed}/    # üretilir, versiyonlanmaz
reports/figures/         # EDA çıktıları
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
