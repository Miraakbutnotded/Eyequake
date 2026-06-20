# EyeQuake — Araştırma Prototipi Bulguları

> Soru: *"Geçmiş deprem kataloğundan deprem gerçekten öngörülebilir mi?"*
> Yöntem: USGS FDSN açık kataloğu (Türkiye, 1990–2026), sıkı out-of-time
> doğrulama, her çıktı naive baseline'lara karşı ölçüldü. Veri konuşur.

## Veri

Proje iki katalogla çalıştı — bu doküman ikisini de içerir; her bölüm hangisini
kullandığını etiketler:
- **USGS FDSN** (20.806 olay, M2.5–7.8): ilk kaynak. Bulgu 0, Track A, Track B (GB)
  ve Track C burada üretildi. Türkiye'de M<4 tamlık sorunu taşıdığı için (↓ Bulgu 0)
  terk edildi.
- **AFAD** (156.689 olay, M≥4.0 = 4.234, b≈1.0 — sağlam): **kanonik** ulusal katalog.
  Track B+ ETAS ve bölge-bazlı ETAS burada üretildi. `scripts/07_fetch_afad.py`.
  (Not: AFAD servisi Türkiye dışı IP'lerden geoblock olabilir — Türkiye içi ağ gerekir.)

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

**Katalog düzeltmesi (2026-06-19):** Bu bölümün ilk sürümü yanlışlıkla **USGS**
kataloğunda (20.806 / 4.741 M≥4) fit edilmişti — AFAD geoblock nedeniyle. Aşağıdaki
tüm sayılar **kanonik AFAD'da yeniden üretildi** (`08_etas.py` + `09_etas_spatial.py`,
hiçbir dosya değiştirilmeden). USGS'te görülen n=0.942 / +40% **kanonik sonuç değildi**.

AFAD'da ulusal havuzlanmış temporal ETAS: **n = 11.517**, beceri **+71.1%** (yukarıdaki
Track B+ tablosuyla birebir). Yani n≈11.5 kanonik veride **aynen geçerli**. Dikkat:
katalog *büyüdükçe* n *yükseliyor* (AFAD, USGS'ten 7.5× büyük → n daha yüksek) — "daha
çok veri → daha fiziksel n" beklentisi bu veride **geçerli değil**. Fark katalog
büyüklüğünden değil **kimliğinden** (USGS≠AFAD: farklı Mc, farklı tetikleme yoğunluğu)
kaynaklanır. n tek bir "doğru" değer değil, izlenmesi gereken bir **tanı göstergesidir**.

### Bölge-bazlı (fay sistemi) ETAS — `scripts/09_etas_spatial.py`

Hipotez: ulusal havuzlanmış fit bağımsız fay sistemlerini (KAF/DAF/Ege/Van) tek
modelde topladığından n fiziksel-olmayan bölgeye itiliyor; bölgelere ayırınca n<1
beklenir. Her bölge ayrı fit edildiğinde (**kanonik AFAD**, M≥4.0, 75/25 train/test):

| Bölge | Olay | b | n (dallanma) | Beceri (climatology'ye) |
|-------|------|---|---------------|--------------------------|
| Ulusal (havuzlanmış) | 4.234 | 0.95 | 11.517 ❌ | +71.1% |
| KAF (Kuzey Anadolu) | 667 | 0.87 | 1.214 ❌ | −13.5% |
| DAF (Doğu Anadolu / 2023) | 1.059 | 0.95 | 5.092 ❌ | **+92.7%** |
| Ege / Helen yayı | 944 | 1.02 | 10.338 ❌ | +21.2% |
| Van | 403 | 1.04 | tanımsız (α≥β) | −1.0% |

**Dürüst değerlendirme (AFAD):** Bölgelere bölmek n<1 sorununu **çözmedi** — kanonik
veride hiçbir bölge fiziksel n vermiyor (hepsi n≥1 ya da tanımsız). USGS'te bazı
bölgelerin n<1 çıkması katalog-spesifik artefakttı. DAF/2023'ün yüksek becerisi
(+92.7%) büyük ölçüde **tek 2023 dizisine overfit** (DAF olaylarının ~%62'si o dizi),
bağımsız tahmin gücü değil. Van: α≥β → n tanımsız + train ~302 olay → model değersiz.
Ayrıca 4 fay kutusu M≥4 olaylarının yalnızca **%72.6'sını** kapsıyor — %27.4'ü (9 adet
M≥6 dahil: Batı/Orta Anadolu grabenleri, Helen yayı doğusu, DAF-KAF eklemi) dışarıda.

**Sonuç:** Bölge-bazlı ETAS yöntemsel olarak doğru yönde ama **henüz güvenilir değil**:
(1) küçük-örneklem bölgeler (KAF train 500, Van 302) MLE'yi dengesizleştiriyor,
(2) fay kutuları sismisitenin ~%27'sini kaçırıyor, (3) n her bölgede ≥1. Fiziksel n
için daha fazla veri + gerçek uzaysal-zamansal ETAS (sadece dikdörtgen kutu değil) +
dizi-bağımsız eşik gerekir. Bölgesel n **tanı göstergesi** olarak izlenmeli, üretim
sinyali olarak değil.

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

## Track D — Zemin/saha kırılganlık (BIS) · ✅ GERÇEK (sınırlı)

Açık yükseklik → eğim → **Vs30** (Wald-Allen) → NEHRP zemin sınıfı + büyütme;
**sıvılaşma** screening proxy; hazard × büyütme = saha-düzeltilmiş risk
(`scripts/10_site_layer.py`, `analysis/site.py`, birim test 6/6).

**Doğrulama — bilinen jeolojiye karşı sanity-check (yer-gerçeği yok):**

| Konum | Vs30 (sınıf) | Büyütme | Sıvılaşma | Beklenti | |
|-------|---|---|---|---|---|
| İstanbul Avcılar (1999 büyütme) | 150 (E) | 2.25× | yüksek | yüksek amp | ✅ |
| Çukurova/Adana deltası | 240 (D) | 1.78× | yüksek | yüksek sıvılaşma | ✅ |
| Erzurum yüksek ova | 620 (C) | 1.11× | çok düşük | düşük | ✅ |
| Hakkari dağlık | 620 (C) | 1.11× | çok düşük | düşük | ✅ |
| Adapazarı (1999 sıvılaşma) | 360 (C) | 1.45× | orta | yüksek | ⚠️ |

4/5 güçlü; Adapazarı (küçük havza) hafif düşük tahmin — 0.25° hücre + ~2 km eğim
penceresi küçük havzaları düzleştiriyor (Wald-Allen ~1 km için kalibre).
**Sınırlar:** Vs30 ~bölgesel çözünürlük (mikrobölgeleme değil); sıvılaşma tarama
göstergesi (sondaj/yağış verisi içermez); deniz hücreleri yumuşak çıkar (maskeleme fast-follow).

## Belirsizlik notu

Skor tablolarındaki nokta-tahminler (örn. +%71.1 beceri) **tek bir 75/25 train/test
bölünmesinden** gelir; tek-fit gürültüsünü gizler. `evaluate.bootstrap_mean_ci` ile
pencere-blok bootstrap %95 CI hesaplanabilir; b-değeri zaten Shi-Bolt standart hatası
(`b_value_aki.sigma`) taşır. Raporlanan rakamlar **±belirsizlik** ile okunmalıdır.

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
   Fetcher kodu hazır (`scripts/07_fetch_afad.py`), ancak AFAD servisi bazı ağlardan
   (örn. Türkiye dışı IP'ler) erişilemiyor olabilir — Türkiye içi bir ağdan
   doğrulanması gerekiyor.
2. **BIS (Bina Bilgi Sistemi)** veri modeli — projenin en savunulabilir, en az kalabalık ayağı.
3. ~~ETAS modeli~~ ✅ (Track B+). ~~Bölge-bazlı ETAS~~ ✅ (`scripts/09_etas_spatial.py`),
   kanonik AFAD'da doğrulandı — ama n her bölgede ≥1 (fiziksel değil). Sıradaki:
   gerçek uzaysal-zamansal ETAS (dikdörtgen kutu değil) + dizi-bağımsız eşik + fay
   kutusu kapsamını genişletme (M≥4'ün ~%27'si dışarıda); harita üstünde bölge-bazlı
   "beklenen oran" katmanı henüz web'e taşınmadı.
4. Hazard + oran sonuçlarını web haritası/dashboard'a taşı (sunum/demo).

## Tekrar üretim

```bash
./.venv/bin/python scripts/01_fetch_catalog.py
./.venv/bin/python scripts/02_eda.py
./.venv/bin/python scripts/03_forecast_baselines.py
./.venv/bin/python scripts/04_hazard_map.py
./.venv/bin/python scripts/05_deterministic_test.py
```
