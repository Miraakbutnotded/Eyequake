# Model Kartı — Bina Kırılganlık Katmanı (BIS Faz-1)

- **Sürüm:** `frag-1.0.0` (`src/eyequake/analysis/fragility.py`)
- **Kaynak tasarım:** `docs/plans/2026-06-28-end-to-end-production-system-design.md` §6.3
- **Durum:** uzman-türetilmiş, **sonuç-doğrulaması yapılmadı**. **Fiyatlama kullanımı kapalı.**

## Ne yapar

Kullanıcı-girişli bina karnesini (yapı tipi, yapım yılı, kat, yumuşak kat, düzensizlik,
çarpışma riski, onarılmamış hasar, imar barışı, güçlendirme) göreli bir kırılganlık
indeksine `V ∈ [0.05, 0.98]`, EMS-98 tarzı sınıfa (A = en kırılgan … F) ve sınırlı bir
çarpana `M_raw ∈ [0.65, 1.80]` çevirir. Çarpan, hazard × saha skorunun (`site_risk`)
üstüne sorgu anında bindirilir: `parcel_risk = clip(round(site_risk · M_eff), 0, 100)`.

## Ne yapmaz

- Hasar/çökme tahmini değildir; mühendislik hasar tespiti / RBTE yerine geçmez.
- Eğitilmiş model değildir — Türkiye için etiketli kırılganlık yer-gerçeği yok.
- Mutlak kayıp kalibrasyonu yoktur; prim fiyatlamasında kullanılamaz (NOT-FOR-PRICING).

## Temel

- `V_base`: yapı tipi × TR deprem yönetmeliği dönemi (pre1975 / 1975–1997 / 1998–2006 /
  2007–2017 / 2018+) matrisi. Bilinmeyen tip/dönem → kohort ortalaması.
- Toplamsal değiştiriciler: kat cezası (rijit olmayan tipler; yığmada >3 kat +0.10),
  yumuşak kat +0.15, düzensizlik +0.08, çarpışma +0.05, onarılmamış hasar +0.12,
  imar barışı +0.10, güçlendirme −0.25 × güven (beyan 0.2, saha/resmi 1.0).
- **Kredibilite büzülmesi:** `κ = tamlık × kaynak güveni` (beyan 0.5, saha 0.85,
  resmi 1.0); `M_eff = 1 + (M_raw − 1)·κ`. `κ < 0.3` → `M_eff = 1` (yalnız saha skoru).
  Eksik bilgi iyimser bir sayıya değil, dürüst saha skoruna geriler.
- **Anti-gaming:** beyan skoru en fazla %10 düşürebilir (`M_eff ≥ 0.90`). `frag-1.0.0`
  parametreleriyle beyanın `κ ≤ 0.5` olması bu sınırı zaten sağlar (en düşük beyan
  `M_eff ≈ 0.91`); taban, parametre değişimine karşı savunma derinliği olarak durur.
  Kötüleştiren beyanlar tam uygulanır.

## Bilinen sınırlar

- Matris ve ağırlıklar bir Türk deprem/yapı mühendisince imzalanmadı — ticari
  kullanımdan önce gözden geçirme ve imza `frag-1.x` sürümüyle kaydedilmeli.
- Sonuç-doğrulaması (2023 Kahramanmaraş hasar-tespit etiketlerine karşı bina-bilgili vs
  yalnız-saha skoru, `evaluate.skill_score` + `bootstrap_mean_ci`) veri erişimine bağlı
  ve henüz yapılmadı. Bu doğrulama baseline'ı yenene kadar fiyatlama kapısı kapalı kalır.
- Faz-1'de çözünürlük hâlâ 0.25° hazard hücresidir; bina karnesi parsel geometrisine
  bağlanmaz (bkz. tasarım §5 "Parcel-granularity vs honesty").
