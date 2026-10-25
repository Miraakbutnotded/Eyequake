"""Negation-aware yasaklı-iddia linter'ı (bilimsel-dürüstlük değişmezi).

Tek doğruluk kaynağı: docs/POSITIONING.md → "Yasaklı kelimeler" + "İstisna: disclaimer".
EyeQuake dışa dönük metinde belirli iddiaları POZİTİF cümle olarak yasaklar, ama AYNI
ifadeler NEGATİF/disclaimer biçiminde ("... DEĞİLDİR", "mümkün değil") KALIR — bunlar
dürüstlüğün kanıtıdır. Bu modül o kuralı prozadan koda taşır; tek-substring assert'lerin
aksine negation-farkındalığı vardır.

Tasarım notları
---------------
* TR İ/ı tuzağı: ``str.casefold()`` İ→i eşlemez (İ → "i̇", combining dot). Eşleşmeden
  ÖNCE İ→i, I→ı pre-map'i yapılır (`_fold`), böylece "TAHMİNİ" → "tahmini".
* Negation penceresi ÇİFT YÖNLÜ: marker ifadeden SONRA ("deprem tahmini DEĞİLDİR") veya
  ÖNCE ("Mümkün değil … Deterministik deprem tahmini" — reddedilen Track C başlığı,
  methodology.html:70) gelebilir. Tek yönlü pencere o dürüst başlığı yanlış işaretlerdi.
"""

from __future__ import annotations

from dataclasses import dataclass

# Track C = deterministik "şu tarih/yer/büyüklük" tahmini: test edilip REDDEDİLDİ.
# Kaynak: reports/FINDINGS.md, CLAUDE.md "Üç Track doktrini".
TRACK_C_STATUS = "rejected"

# docs/POSITIONING.md "Yasaklı kelimeler (dışa dönük metinde POZİTİF iddia olarak)":
#   - "deprem tahmini / öngörüsü"  (gelecek depremin tarih/yer/büyüklük vaadi)
#   - "yapay sinir ağları / neural network ile tahmin"
#   - "ne zaman / nerede / kaç büyüklüğünde deprem"
#   - "erken uyarı sistemi"  (sensör tabanlı, bizde yok)
# Aşağıdaki ifadeler bu listenin pozitif-iddia çekirdeğidir; folded substring olarak
# eşlenir. "yapay sinir ağı ile deprem tahmini" gibi neural-net iddiaları "deprem tahmini"
# üzerinden yakalanır. Çift-sayımı önlemek için ifadeler ayrık tutulur.
BANNED_CLAIMS = (
    "deprem tahmini",          # POSITIONING: "deprem tahmini / öngörüsü"
    "deprem öngörüsü",         # POSITIONING: "… / öngörüsü"
    "erken uyarı",             # POSITIONING: "erken uyarı sistemi"
    "ne zaman deprem",         # POSITIONING: "ne zaman … deprem"
    "nerede deprem",           # POSITIONING: "nerede … deprem"
    "kaç büyüklüğünde deprem",  # POSITIONING: "kaç büyüklüğünde deprem"
)

# POSITIONING "İstisna: disclaimer / negatif / test-ettik-reddettik" markerları.
# İ/I içermez → _fold bunlar için kimliktir; folded pencereye doğrudan uygulanabilir.
NEGATION_MARKERS = (
    "mümkün değil",
    "mumkun degil",
    "değil",
    "degil",
    "yapmaz",
)

# Negation marker'ın ifadeye uzaklık penceresi (her iki yön, ~karakter).
NEGATION_WINDOW = 40


@dataclass(frozen=True)
class ClaimHit:
    """Pozitif (negate edilmemiş) yasaklı-iddia bulgusu."""

    phrase: str
    index: int


def _fold(s: str) -> str:
    """Türkçe-güvenli casefold: İ→i, I→ı pre-map'i + casefold.

    Python ``str.casefold()`` İ (U+0130) için "i̇" (i + combining dot) üretir; bu sade
    "i" ile eşleşmez. Eşleşmeden önce pre-map ile bu tuzak kapatılır.
    """
    return s.replace("İ", "i").replace("I", "ı").casefold()


def _is_negated(folded: str, start: int, end: int) -> bool:
    """İfade [start:end] çevresindeki çift-yönlü pencerede negation marker var mı?

    Marker ifadeden sonra (disclaimer "… DEĞİLDİR") veya önce (reddedilen-Track-C
    başlığı "Mümkün değil … deprem tahmini") gelebilir.
    """
    lo = max(0, start - NEGATION_WINDOW)
    hi = min(len(folded), end + NEGATION_WINDOW)
    window = folded[lo:hi]
    return any(marker in window for marker in NEGATION_MARKERS)


def scan_text(text: str) -> list[ClaimHit]:
    """Metni tara, negate EDİLMEMİŞ yasaklı iddiaları ClaimHit listesi olarak döndür.

    Boş liste = pozitif yasaklı-iddia yok (disclaimer/negatif kullanımlar serbesttir).
    """
    folded = _fold(text)
    hits: list[ClaimHit] = []
    for phrase in BANNED_CLAIMS:
        needle = _fold(phrase)
        pos = folded.find(needle)
        while pos != -1:
            end = pos + len(needle)
            if not _is_negated(folded, pos, end):
                hits.append(ClaimHit(phrase=phrase, index=pos))
            pos = folded.find(needle, pos + 1)
    hits.sort(key=lambda h: h.index)
    return hits
