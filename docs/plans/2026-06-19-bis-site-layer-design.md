# BIS — Zemin/Saha Kırılganlık Katmanı (v1) — Tasarım

> Durum: tasarım onaylandı (brainstorming, 2026-06-19). İnşa bekliyor.
> Amaç: EyeQuake'in ticari farklılaştırıcısı olan BIS'in (Bina Bilgi Sistemi)
> ilk ayağı — **bina verisi gerektirmeyen** zemin/saha kırılganlık katmanı.
> Bina-seviyesi (kullanıcı karnesi / resmi kayıt) sonraki aşamalar.

## Karar özeti

| Karar | Seçim | Gerekçe |
|-------|-------|---------|
| BIS v1 yaklaşımı | Zemin/saha katmanı (önce) | Bina verisi bloker; zemin açık veriyle yapılır |
| Kapsam | Amplifikasyon + sıvışlaşma + birleşik saha-risk | Kullanıcı "hepsi" dedi |
| Vs30 kaynağı | Eğimden türet (Wald-Allen 2007), USGS raster değil | Dependency-light, batch-dostu, aynı bilim |
| Sıvılaşma | Screening proxy (jeoteknik değil) | Sondaj/yeraltı suyu verisi yok — dürüst sınır |
| Derin öğrenme | Site layer'a DEĞİL | Etiketli veri yok; DL ayrı forecasting track'i |

## 1. Veri kaynakları (doğrulandı)

- **Yükseklik → eğim → Vs30:** open-meteo elevation API (anahtarsız, **batch** — doğrulandı:
  Ankara 848 m, 4 nokta tek çağrı). Her grid hücresi + komşularının yüksekliği → eğim →
  Vs30 (Wald & Allen 2007 slope-Vs30, aktif tektonik bins).
- **Sıvılaşma prediktörleri:** Vs30, yükseklik/eğim, su yakınlığı, yıllık yağış
  (open-meteo archive — doğrulandı). Zhu et al. tarzı screening.
- **Birleşik saha-risk:** mevcut hazard indeksi (AFAD sismisitesi, 0–100) × amplifikasyon.
- Hepsi açık, anahtarsız, backend yok, global (AFAD gibi geoblock riski yok).
- *Reddedilen alternatif:* USGS Vs30 raster grid (HTTP 200 ama rasterio/GDAL ister, ağır).

## 2. Hesaplama

**① Zemin büyütmesi** — Eğim → Vs30 (Wald-Allen, aktif tektonik):

| Eğim (m/m) | Vs30 (m/s) | NEHRP |
|---|---|---|
| <3e-4 | 180 | E |
| 3e-4–3.5e-3 | 240 | D |
| 3.5e-3–0.018 | 300–360 | D |
| 0.018–0.05 | 490 | C |
| 0.05–0.10 | 620 | C |
| >0.10 | 760 | B |

Amplifikasyon (Borcherdt tarzı, ref Vs30=760): `Amp = (760/Vs30)^0.5` →
E ~2.1×, D ~1.4–1.6×, C ~1.1–1.25×, B ~1.0×.

**② Sıvılaşma yatkınlığı** (screening proxy): 4 prediktör (düşük Vs30 [en ağır],
düşük yükseklik+düz eğim, su yakınlığı, yağış) 0–1 puana → ağırlıklı topla →
sınıf (çok düşük / düşük / orta / yüksek). **Birleşik skora karıştırılmaz** —
ayrı tehlike türü, ayrı bayrak.

**③ Birleşik saha-risk** = hazard_index × normalize(amplifikasyon), 0–100'e ölçekle.
"Nerede deprem olur (hazard) × zemin nasıl tepki verir (amplifikasyon)".

Hepsi 0.25° grid hücresi başına önceden hesaplanır (mevcut hazard grid hücreleriyle
birebir hizalı).

## 3. Mimari

**Yeni Python:**
- `src/eyequake/data/elevation.py` — open-meteo batch yükseklik çekici + `data/raw` cache.
- `src/eyequake/analysis/site.py` — `slope_to_vs30()`, `amplification()`,
  `liquefaction_susceptibility()`, `site_adjusted_risk()` (saf, test edilebilir).
- `scripts/10_site_layer.py` — yükseklik grid çek → metrik hesapla → GeoJSON yaz.

**Veri akışı:** Mevcut `web/data/hazard_cells.geojson` aynı hücreleri zenginleştirilir
(`vs30, site_class, amp, liq_class, site_risk` props). Tek tıkla hem hazard hem zemin.

**Web (app.js):** Katman seçici butonlar `Hazard | Amplifikasyon | Sıvılaşma | Saha-risk`
(aynı GeoJSON, farklı `style`). Panel'e zemin satırları + ayrı sıvılaşma bayrağı + lejant.

**Performans:** ~1886 hücre × ~5 nokta ≈ 9k yükseklik sorgusu → ~100'lük batch ~90 çağrı,
cache'li. Tek seferlik üretim → web backend'siz kalır.

**Dokunulmayan:** hazard/ETAS/forecast/AFAD pipeline — sadece ekleme.

## 4. Dürüstlük & test

- **Çerçeveleme:** amplifikasyon = standart; sıvılaşma = **"tarama göstergesi, jeoteknik
  değil; sondaj verisi yok"** etiketi zorunlu; saha-risk = "göreli/tanımlayıcı".
  "Deprem tahmini değildir" duruşu korunur.
- **Doğrulama (yer-gerçeği yok → bilinen jeoloji):** yüksek amplifikasyon çıkmalı —
  Adapazarı, Ergene/Trakya, İzmit kıyısı, Çukurova, Gediz/B.Menderes graben, İstanbul
  Avcılar. Düşük — dağlık D. Anadolu/Toroslar. Sıvılaşma "yüksek" → Adapazarı + kıyı
  alüvyonu + deltalar. Vs30 dağılımı yayımlanmış TR aralıklarıyla kıyas.
- **Birim testler:** `slope_to_vs30` (sınır eğimleri), `amplification` (Vs30'da monoton),
  liquefaction (prediktörlerde monoton), birleşik skor; bilinen şehir spot-check.
- **Belirsizlik notu (UI):** "~1 km çözünürlük; slope-türetilmiş Vs30, mikrobölgeleme
  yerine geçmez."

## Gelecek araştırma — DL forecasting track (ayrı)

Site layer'a DEĞİL. Temporal deprem forecasting'e (Track B+): **neural point process /
Neural-Hawkes / deep-ETAS**, ETAS'a karşı **dürüst benchmark**. Uyarı: TR kataloğu
küçük + durağansız → DL muhtemelen ETAS'ı yenmez; sonuç (yensin/yenmesin) raporlanır.
Pitch'teki "AI" hikayesi buradan beslenir — site layer'da sahte DL'den kaçınılır.

## Sonraki BIS aşamaları (bu tasarımın dışında)

1. Kullanıcı-girişli bina karnesi (yaş/kat/yapı tipi → kırılganlık sınıfı).
2. Resmi bina verisi (tapu/belediye/kentsel dönüşüm) — erişim/izin gerektirir.
