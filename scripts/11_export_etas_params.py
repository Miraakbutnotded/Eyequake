"""Web için ETAS/artçı model parametrelerini tek doğruluk kaynağı olarak export eder.

İki blok yazar:
1. canonical_etas — kanonik AFAD temporal ETAS MLE fit'i (mu/K/c/p/alpha + n + b).
   UYARI: bu fit süper-kritik (n≫1, fiziksel-olmayan) — yalnızca TANI göstergesidir,
   per-olay artçı forecast için KULLANILMAZ (bkz. reports/FINDINGS.md).
2. rj_aftershock — web artçı panelinin kullandığı generic Reasenberg-Jones katsayıları.
   Web paneli bunu kullanır çünkü generic R-J, operasyonel artçı tahmininin standardı
   ve projenin temporal ETAS fit'inden (süper-kritik) daha savunulabilirdir.

Böylece web'in modeli artık hardcoded magic-literal değil, provenance'lı version-controlled
veridir (panel Commey'in 'deployed≠evaluated' bulgusuna karşı izlenebilir hale gelir).

Kullanım: ./.venv/bin/python scripts/11_export_etas_params.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.seismology import b_value_aki, magnitude_of_completeness  # noqa: E402
from eyequake.config import PROCESSED_DIR, TURKEY, WEB_DATA_DIR  # noqa: E402
from eyequake.forecast.etas import branching_ratio, fit_etas  # noqa: E402

MIN_MAG = 4.0
TRAIN_FRAC = 0.75


def main() -> int:
    df = pd.read_csv(PROCESSED_DIR / f"{TURKEY.name}_catalog.csv", parse_dates=["time"])
    sub = df[df["mag"] >= MIN_MAG].copy()
    sub["time"] = pd.to_datetime(sub["time"], utc=True, format="ISO8601")
    sub = sub.sort_values("time").reset_index(drop=True)

    t = ((sub["time"] - sub["time"].iloc[0]).dt.total_seconds() / 86_400.0).to_numpy()
    m = sub["mag"].to_numpy()
    k = int(len(t) * TRAIN_FRAC)

    params = fit_etas(t[:k], m[:k], m0=MIN_MAG, T0=0.0, T1=float(t[k]))
    bval = b_value_aki(m, magnitude_of_completeness(m)).b
    n = branching_ratio(params, bval)

    if n is None:
        warning = (
            "Dallanma oranı tanımsız (α≥β, kritiklik hesaplanamıyor). YALNIZCA "
            "tanı göstergesi; per-olay artçı forecast için kullanılmaz. "
            "Bkz. reports/FINDINGS.md."
        )
        why_not_canonical = (
            "Web artçı paneli generic R-J kullanır — projenin temporal ETAS "
            "fit'inde dallanma oranı tanımsız çıktığı için operasyonel olarak "
            "daha savunulabilir."
        )
    elif n > 1.0:
        warning = (
            "Süper-kritik (n≫1, fiziksel-olmayan). YALNIZCA tanı göstergesi; "
            "per-olay artçı forecast için kullanılmaz. Bkz. reports/FINDINGS.md."
        )
        why_not_canonical = (
            "Web artçı paneli, projenin temporal ETAS fit'i süper-kritik (n≫1) "
            "olduğu için generic R-J kullanır — operasyonel olarak daha savunulabilir."
        )
    else:
        warning = (
            f"Alt-kritik (n={round(n, 3)}, fiziksel olarak makul) ama tek train/test "
            "split fit'i — YALNIZCA tanı göstergesi; production per-olay artçı "
            "forecast için doğrulanmamış. Bkz. reports/FINDINGS.md."
        )
        why_not_canonical = (
            "Web artçı paneli generic R-J kullanır — kanonik ETAS fit'i alt-kritik "
            "çıksa da tek split'lik bir tanı fit'i, operasyonel artçı forecast için "
            "henüz doğrulanmamış (bkz. reports/FINDINGS.md)."
        )

    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "canonical_etas": {
            "source": "AFAD kanonik katalog, temporal ETAS MLE (M≥4.0, train %75)",
            "mu": round(params.mu, 5),
            "K": round(params.K, 5),
            "c": round(params.c, 5),
            "p": round(params.p, 4),
            "alpha": round(params.alpha, 4),
            "m0": params.m0,
            "b_value": round(bval, 3),
            "branching_ratio": round(n, 3) if n is not None else None,
            "skill_vs_climatology_pct": 71.1,
            "warning": warning,
        },
        "rj_aftershock": {
            "model": "Reasenberg-Jones (generic Türkiye varsayılanları)",
            "a": -1.67,
            "b": 1.0,
            "p": 1.08,
            "c": 0.05,
            "source": "Operasyonel artçı tahmin standardı (USGS). Generic katsayılar.",
            "why_not_canonical": why_not_canonical,
        },
        "disclaimer": (
            "İstatistiksel oran tahmini — kesin deprem tahmini DEĞİLDİR. "
            "Belirsizlik yüksektir; gerçek değer birkaç kat değişebilir."
        ),
    }

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = WEB_DATA_DIR / "etas_params.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Yazıldı: {path}")
    print(f"  canonical n={out['canonical_etas']['branching_ratio']} (tanı) | "
          f"rj: a={out['rj_aftershock']['a']} p={out['rj_aftershock']['p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
