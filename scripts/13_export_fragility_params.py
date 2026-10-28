"""BIS Faz-1 bina kırılganlık parametrelerini web için export eder.

`analysis/fragility.py` tek doğruluk kaynağıdır; statik web (backend yok) aynı formülü
`web/fragility.js`'te bu JSON'la koşar. Altın vektörler JSON'a gömülür — JS yüklemede
onları yeniden hesaplar, sapma varsa bina karnesi formunu kapatır (fail-closed).
`tests/test_fragility_web.py` JSON'un koddan bayatlamadığını CI'da korur.

Katalog/ağ gerektirmez. Kullanım: python scripts/13_export_fragility_params.py
"""

from __future__ import annotations

import json

from eyequake.analysis.fragility import export_params
from eyequake.config import WEB_DATA_DIR


def main() -> int:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA_DIR / "fragility_params.json"
    out = export_params()
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Yazıldı: {path}")
    print(f"  {out['fragility_model_version']} | altın vektör: {len(out['golden'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
