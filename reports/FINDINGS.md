# EyeQuake — Araştırma Prototipi Bulguları

> Soru: *"Geçmiş deprem kataloğundan deprem gerçekten öngörülebilir mi?"*
> Yöntem: USGS FDSN açık kataloğu (Türkiye, 1990–2026), sıkı out-of-time
> doğrulama, her çıktı naive baseline'lara karşı ölçüldü. Veri konuşur.

## Veri

- 20.806 olay, 1990-01 → 2026-06, M2.5–7.8 (M7.8 = 6 Şubat 2023 Kahramanmaraş).
- Kaynak: USGS FDSN Event API (açık, anahtarsız). Otoriter alternatif: AFAD/Kandilli.

## Bulgu 0 — Katalog tamlığı (kritik ön koşul)

USGS, Türkiye'de küçük depremleri **zaman-değişken** raporluyor:

| Eşik | 1994–2003 | 2014–2023 |
|------|-----------|-----------|
| M≥2.5 | ~852/yıl | ~177/yıl |
| M≥4.0 | ~87/yıl | ~171/yıl |

M≥2.5'teki %80 düşüş deprem azalması değil, ~2009 raporlama rejimi değişimidir.
- Magnitude of completeness Mc ≈ **2.95**; Gutenberg-Richter b ≈ **0.53** (tipik 0.8–1.1'in
  altında → katalog heterojenliği belirtisi).
- **Karar: modelleme eşiği M≥4.0** (4.741 olay, tüm dönem tutarlı/tam). M≥2.5'te
  eğitmek sismisiteyi değil raporlamayı modellerdi.

## Track A — Tehlike (hazard) haritası · ✅ GERÇEK

0.25° ızgarada yıllık M≥4.0 oranı (PSHA'nın a-value alanı). 913 dolu hücre, 36.4 yıl.
En yüksek oranlı bölgeler bilinen faylarla örtüşüyor: Doğu Anadolu Fayı (2023 zonu),
Kuzey Anadolu Fayı, Ege/Helen yayı, Van. → `reports/figures/06_hazard_rate_map.png`

**Değerlendirme:** Standart, savunulabilir, ürünleştirilebilir. Olasılıksızdır
("şu bölge yılda kaç deprem üretir"), bu yüzden iddialı değildir ve doğrudur.

## Track B — Olasılıksal oran öngörüsü · ⚠️ ZAYIF–ORTA, GERÇEK

30g pencerede M≥4.0 olay sayısı, out-of-time test:

| Model | MAE | Poisson Dev | Climatology'ye beceri |
|-------|-----|-------------|------------------------|
| climatology (Poisson background) | 12.92 | 47.01 | — |
| persistence | 17.42 | 43.17 | +8.2% |
| **gradient boosting** | 13.01 | **39.76** | **+15.4%** |
| poisson GLM | 23.37 | 57.10 | −21.5% |

**Değerlendirme:** Gerçek ama mütevazı sinyal. Kazanç **artçı kümelenmesinden**
(Omori/ETAS) geliyor — "aktif dönem devam eder". MAE'de fark yok (model kazancı
yüksek-aktivite pencerelerinde yoğun). Bu deprem tahmini değil, *oran* öngörüsü.

### Track B+ — ETAS (fiziksel-temelli olasılıksal model)

Temporal ETAS (μ + Omori-Utsu tetikleme), MLE ile fit (M≥4.0, train 75%):

| Model | MAE | Poisson Dev | Climatology'ye beceri |
|-------|-----|-------------|------------------------|
| **ETAS** | **12.86** | **22.15** | **+71.1%** |
| climatology | 19.53 | 76.52 | — |
| persistence | 30.56 | 88.54 | −15.7% |

ETAS hem deviance hem MAE'de baseline'ı açık ara yener (gradient boosting +%15 idi).
Forecastlanabilen sinyal kesinlikle artçı tetiklenmesi.

**Dürüst uyarı:** Fit edilen parametreler fiziksel olarak temiz değil — dallanma
oranı n≈11.5 (n<1 olmalı; n≥1 = sönmeyen kaskad), p≈1.01 alt sınırda. Sebep:
temporal-only, tüm-Türkiye ETAS bağımsız bölgeleri (KAF/DAF/Ege) tek havuzda
topluyor + 2023 dizisi tetiklemeyi şişiriyor. **Forecasting becerisi gerçek;
parametre yorumu için uzaysal-zamansal / bölge-bazlı ETAS gerekir.**

## Track C — Deterministik tahmin (büyüklük) · ❌ MÜMKÜN DEĞİL

Bir sonraki olayın büyüklüğü, önceki 10 olaydan:

| Model | MAE | R² |
|-------|-----|-----|
| climatology (ortalama) | 0.227 | −0.005 |
| persistence (son M) | 0.303 | −0.921 |
| gradient boosting | 0.230 | −0.019 |

**Değerlendirme:** R² ≈ 0 — büyüklük geçmişten **tahmin edilemiyor**. Gutenberg-Richter'in
beklentisi (büyüklükler ~bağımsız üstel dağılım). ML "ortalamayı söyle"yi bile yenemiyor.

İkili "gelecek 30g'de M≥5?" görevi: ML Brier 0.276 vs base-rate 0.275 (beceri −0.2%),
ROC-AUC 0.551 ≈ rastgele. Train→test base-rate %41→%59 (2023 durağansızlığı).
→ base-rate üstünde anlamlı sinyal yok.

## Sonuç — ürün için ne anlama geliyor

| İddia | Statü | Ürün konumu |
|-------|-------|-------------|
| "Hangi bölge riskli?" (hazard) | ✅ Sağlam | **Çekirdek ürün** |
| "Yakın dönemde aktivite artar mı?" (oran/artçı) | ⚠️ Mütevazı, gerçek | Yardımcı katman |
| "Şu tarihte/yerde/büyüklükte deprem" (deterministik) | ❌ Desteksiz | **Vaat etme** |

EyeQuake'in gerçek değeri **sismik risk + bina kırılganlık zekası** (BIS) +
artçı/oran öngörüsünde. "Deprem tahmincisi" çerçevesi kendi verimizce çürüyor;
dürüst çerçeve hem bilimsel hem ticari olarak daha güçlü.

## Sonraki adımlar (öneri)

1. **AFAD/Kandilli kataloğu** entegrasyonu → daha düşük Mc, daha çok olay, daha sağlam Track A/B.
2. **BIS (Bina Bilgi Sistemi)** veri modeli — projenin en savunulabilir, en az kalabalık ayağı.
3. ~~ETAS modeli~~ ✅ yapıldı (Track B+). Sıradaki: **uzaysal-zamansal / bölge-bazlı
   ETAS** (fiziksel n<1 parametreleri + harita üstünde "beklenen oran" katmanı).
4. Hazard + oran sonuçlarını web haritası/dashboard'a taşı (sunum/demo).

## Tekrar üretim

```bash
./.venv/bin/python scripts/01_fetch_catalog.py
./.venv/bin/python scripts/02_eda.py
./.venv/bin/python scripts/03_forecast_baselines.py
./.venv/bin/python scripts/04_hazard_map.py
./.venv/bin/python scripts/05_deterministic_test.py
```
