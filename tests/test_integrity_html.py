"""Served HTML dosyalarında pozitif yasaklı-iddia taraması.

scan_text, negation-aware linter'dır: "deprem tahmini DEĞİLDİR" gibi disclaimer
kullanımları geçerlidir ve işaretlenmez. Yalnızca negate EDİLMEMİŞ pozitif iddialar
(örn. "Sistemimiz deprem tahmini yapar") hata olarak raporlanır.

Her push'ta CI'da çalışır — web/data/*.json'un aksine HTML dosyaları git'te takip
edilir, dolayısıyla harici katalog gerekmez.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eyequake.integrity import scan_text

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
HTML_FILES = ["index.html", "methodology.html"]


@pytest.mark.parametrize("filename", HTML_FILES)
def test_html_contains_no_positive_banned_claims(filename: str) -> None:
    """Served HTML must not contain positive (non-negated) banned claims.

    Negated or disclaimer uses (e.g. "deprem tahmini DEĞİLDİR", "Mümkün değil …
    Deterministik deprem tahmini") are allowed and expected — they are the honesty
    layer. This test catches accidentally added positive claims (e.g. copy that
    promises earthquake prediction or early-warning capability).
    """
    path = WEB_ROOT / filename
    if not path.exists():
        pytest.skip(f"HTML dosyası bulunamadı: {path}")

    text = path.read_text(encoding="utf-8")
    hits = scan_text(text)

    assert hits == [], (
        f"{filename}: pozitif yasaklı iddia bulundu — {[h.phrase for h in hits]}. "
        "Disclaimer/negatif kullanımlar (DEĞİLDİR, değil) serbesttir; "
        "yalnızca negate edilmemiş pozitif iddialar hata sayılır."
    )
