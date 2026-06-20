# EyeQuake — Konumlandırma (kanonik tek doğruluk kaynağı)

> Tüm dışa dönük materyaller (README, web, deck, canvas) **bu dosyaya** hizalanır.
> Sapma olursa kazanan bu dosyadır.
> **DURUM: sigorta/reasürans varsayılanıyla taslak — founder onayı bekliyor** (Bora/Alp).

## Tek-cümle JTBD (beachhead)

> *"Bir konut/portföy sigortacısı olarak, bir parselin sismik + bina-kırılganlık
> riskini fiyatlamak istiyorum ki primi ve teminat limitini doğru belirleyebileyim."*

## Beachhead müşteri (TEK)

**Sigorta / reasürans** (parsel-risk → fiyatlama). Veri-alıcı kültürü olan, ödeme
gücü ve net karar anı (underwriting) bulunan tek segment.

**Reddedilen / park edilen segmentler** (şimdilik DEĞİL — kayıt için):
inşaat şirketleri · belediye/kentsel dönüşüm · afet kurumu · son kullanıcı/halk.
*(Panel konsensüsü: "herkesi hedefleme" fatal flag'di. Tek Job seç, gerisini park et.)*

## One-liner (pozisyon)

> **EyeQuake — parsel-bazlı sismik risk + bina kırılganlık istihbaratı.**
> Sismisite hazard'ı (nerede) ile zemin/bina kırılganlığını (nasıl tepki verir)
> birleştirip sigortacıya parsel-seviyesi risk skoru verir.

- **Hazard = commodity** (sismisite oranı — herkesin erişebildiği açık veri).
- **BIS / bina-kırılganlık = çekirdek farklılaştırıcı** (rakip SeismicAI'de yok).
- **Olasılıksal oran/artçı = yardımcı katman** (deprem tahmini DEĞİL).

## Yasaklı kelimeler (dışa dönük metinde POZİTİF iddia olarak)

- "deprem tahmini / öngörüsü" (gelecek bir depremin **tarih/yer/büyüklük** vaadi)
- "yapay sinir ağları / neural network ile tahmin"
- "ne zaman / nerede / kaç büyüklüğünde deprem"
- "erken uyarı sistemi" (sensör tabanlı, bizde yok)

> **İstisna:** disclaimer / negatif / "test ettik-reddettik" bağlamı KALIR —
> örn. "deprem tahmini DEĞİLDİR", "deterministik tahrip mümkün değil (Track C)".
> Bunlar dürüstlüğün kanıtı, yasak değil.

## İzinli çerçeve

- "göreli sismik risk indeksi (0–100)" · "saha-düzeltilmiş risk"
- "beklenen sismisite oranı / artçı aktivite olasılığı (istatistiksel)"
- "bina kırılganlık / zemin büyütme istihbaratı"
- "gözlemsel risk istihbaratı — tahmin değil"

## Hizalama checklist

| Materyal | one-liner eşit | beachhead eşit | JTBD eşit |
|---|---|---|---|
| README.md | ⬜ | ⬜ | ⬜ |
| web (index/app) | ⬜ | ⬜ | ⬜ |
| methodology.html | ⬜ | ⬜ | ⬜ |
| Pitch deck (PDF) | ⬜ (founder) | ⬜ | ⬜ |
| Lean/Business Canvas | ⬜ (founder) | ⬜ | ⬜ |
