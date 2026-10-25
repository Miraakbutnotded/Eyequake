"""Integrity modülü testleri — negation-aware yasaklı-iddia linter'ı (afet-güvenlik kritik).

EyeQuake'in bilimsel-dürüstlük konumlandırması (docs/POSITIONING.md) belirli dışa
dönük iddiaları POZİTİF cümle olarak yasaklar ("deprem tahmini" vb.), ama AYNI ifadeler
NEGATIF/disclaimer biçiminde ("... DEĞİLDİR") KALMALIDIR. Bu testler o ayrımı koda döker.

Çalıştır: ./.venv/bin/python -m pytest tests/test_integrity.py -v
"""

from __future__ import annotations

from eyequake.integrity import TRACK_C_STATUS, scan_text


def test_track_c_is_rejected_constant():
    # Track C (deterministik tahmin) test edilip reddedildi — reports/FINDINGS.md.
    assert TRACK_C_STATUS == "rejected"


def test_positive_banned_claim_is_flagged():
    hits = scan_text("Sistemimiz deprem tahmini yapar ve sizi uyarır.")
    assert hits, "pozitif yasaklı-iddia işaretlenmeli"
    assert any("deprem tahmini" in h.phrase for h in hits)


def test_negated_disclaimer_is_allowed():
    hits = scan_text("Bu bir deprem tahmini DEĞİLDİR; göreli risk indeksidir.")
    assert hits == [], "negatif/disclaimer bağlamı yasak değildir"


def test_turkish_casefold_uppercase_claim_flagged():
    # TR İ/ı tuzağı: str.casefold() İ→i eşlemez; pre-map gerekir.
    hits = scan_text("DEPREM TAHMİNİ MOTORU")
    assert len(hits) >= 1, "büyük harfli iddia da yakalanmalı (İ casefold)"


def test_neural_net_prediction_flagged():
    hits = scan_text("yapay sinir ağı ile deprem tahmini")
    assert hits, "neural-net ile tahmin iddiası işaretlenmeli"


def test_clean_text_no_hits():
    hits = scan_text("Göreli sismik risk indeksi (0–100), tanımlayıcı bir göstergedir.")
    assert hits == [], "izinli çerçeve iddia içermez"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} geçti")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
