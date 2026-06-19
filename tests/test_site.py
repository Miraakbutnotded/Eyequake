"""site.py birim testleri — düz assert (pytest gerekmez).

Çalıştır: ./.venv/bin/python tests/test_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.site import (  # noqa: E402
    amplification,
    compute_slope,
    liquefaction,
    nehrp_class,
    site_adjusted_risk,
    slope_to_vs30,
)


def test_slope_to_vs30_bounds_and_monotonic():
    assert slope_to_vs30(0.0) == 150        # çok düz → yumuşak
    assert slope_to_vs30(0.30) == 760       # dik → kaya
    vs = [slope_to_vs30(s) for s in [0, 1e-3, 5e-3, 0.02, 0.07, 0.2]]
    assert vs == sorted(vs), f"Vs30 eğimle monoton artmalı: {vs}"


def test_nehrp_class():
    assert nehrp_class(150) == "E"
    assert nehrp_class(300) == "D"
    assert nehrp_class(490) == "C"
    assert nehrp_class(760) == "B"
    assert nehrp_class(1600) == "A"


def test_amplification_monotonic_decreasing():
    assert amplification(760) == 1.0                 # referans kaya
    assert amplification(150) > amplification(760)    # yumuşak daha çok büyütür
    amps = [amplification(v) for v in [150, 300, 490, 620, 760]]
    assert amps == sorted(amps, reverse=True), f"amp Vs30 ile azalmalı: {amps}"
    assert amplification(150) > 2.0                   # E sınıfı güçlü büyütme


def test_compute_slope():
    assert compute_slope(100, 100, 100, 100, 39.0, 0.02) == 0.0   # düz
    s = compute_slope(200, 100, 100, 100, 39.0, 0.02)             # doğuya eğimli
    assert s > 0


def test_liquefaction_extremes_and_monotonic():
    score_soft, cls_soft = liquefaction(vs30=150, elevation=10, slope=0.0005)
    score_rock, cls_rock = liquefaction(vs30=760, elevation=1800, slope=0.30)
    assert cls_soft == "yüksek", f"yumuşak ova yüksek olmalı: {cls_soft}"
    assert cls_rock == "çok düşük", f"dik kaya çok düşük olmalı: {cls_rock}"
    # Vs30 düştükçe skor artmalı (diğerleri sabit)
    s_hi, _ = liquefaction(vs30=700, elevation=50, slope=0.001)
    s_lo, _ = liquefaction(vs30=200, elevation=50, slope=0.001)
    assert s_lo > s_hi


def test_site_adjusted_risk():
    base = 50
    assert site_adjusted_risk(base, amp=1.15) == 50          # nötr (amp_ref)
    assert site_adjusted_risk(base, amp=2.0) > base          # büyütme artırır
    assert site_adjusted_risk(base, amp=1.0) < base          # kaya azaltır
    assert site_adjusted_risk(95, amp=2.5) == 100            # 100'e kırpılır


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
