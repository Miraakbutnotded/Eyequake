"""Web dürüstlük/güvenlik değişmezleri (statik kontrol) — afet-güvenlik kritik.

Bu testler, expert panelinin işaretlediği dürüstlük güvencelerinin web'de kalıcı
olduğunu korur. Çalıştır: ./.venv/bin/python tests/test_web_honesty.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_disclaimer_present():
    assert "deprem tahmini değildir" in _read("index.html").lower() or \
        "deprem tahmini <strong>değildir" in _read("index.html").lower()


def test_mobile_does_not_hide_disclaimer():
    css = _read("style.css")
    # @media bloğunda disclaimer-pill için display:none OLMAMALI
    mobile = css[css.find("@media"):] if "@media" in css else ""
    assert not re.search(r"\.disclaimer-pill[^}]*display:\s*none", mobile), \
        "disclaimer mobilde gizlenmemeli (afet-güvenlik)"


def test_rj_loaded_from_json_not_hardcoded():
    app = _read("app.js")
    assert "etasParams.rj_aftershock" in app or "rj_aftershock" in app, \
        "RJ katsayıları etas_params.json'dan yüklenmeli (provenance)"


def test_uncertainty_band_not_point_estimate():
    app = _read("app.js")
    assert "etasBand" in app, "ETAS sayıları belirsizlik aralığı göstermeli"
    assert "kesin sayı değildir" in app, "sahte-kesinlik uyarısı olmalı"


def test_false_reassurance_guard():
    app = _read("app.js")
    assert "güvenli demek DEĞİLDİR" in app or "GÜVENLİ demek DEĞİLDİR" in app, \
        "düşük gözlem yanlış-güven uyarısı olmalı"


def test_methodology_page_exists_and_honest():
    m = _read("methodology.html")
    for term in ["deterministik", "Mümkün değil", "tarama göstergesi", "AFAD"]:
        assert term in m, f"methodology.html '{term}' içermeli"


def test_etas_panel_not_called_etas():
    # Generic R-J kullanıldığı için panel başlığı 'ETAS' iddia etmemeli
    assert "Artçı Aktivite Beklentisi" in _read("index.html")


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
