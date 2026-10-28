"""Web bina karnesi ↔ Python fragility paritesi (BIS Faz-1).

1. web/data/fragility_params.json koddan bayatlamamalı (export_params ile birebir).
2. web/fragility.js geniş bir karne ızgarasında Python ile birebir aynı sonucu vermeli
   (node yoksa atlanır; JS ayrıca yüklemede altın vektörlerle kendini denetler).

Çalıştır: ./.venv/bin/python -m pytest tests/test_fragility_web.py -v
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from eyequake.analysis.fragility import (
    STRUCTURE_TYPES,
    BuildingCard,
    apply_building_overlay,
    export_params,
)
from eyequake.integrity import scan_text

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "web" / "data" / "fragility_params.json"
YEAR = 2026

_NODE_RUNNER = """
const fs = require("fs");
const F = require(process.argv[2]);
const P = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const cases = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const m = F.create(P);
const out = cases.map((c) => {
  const o = m.overlay(c.site_risk, c.card, c.year);
  if (o.errors.length) return { errors: o.errors.map((e) => e[0]) };
  const a = o.assessment;
  return {
    parcel_risk: o.parcel_risk, resolution: o.resolution,
    multiplier_effective: o.multiplier_effective, downward_capped: o.downward_capped,
    v: a.vulnerability_index, cls: a.vulnerability_class, m_raw: a.multiplier_raw,
    kappa: a.kappa, warnings: a.plausibility_warnings, imputed: a.imputed_fields,
  };
});
process.stdout.write(JSON.stringify({ selfcheck: m.selfCheck(), out }));
"""


def test_served_params_match_code():
    served = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert served == json.loads(json.dumps(export_params())), (
        "fragility_params.json bayat — scripts/13_export_fragility_params.py koş"
    )


def test_served_params_copy_is_honest():
    p = export_params()
    assert "tahmini" in p["disclaimer"].lower()
    assert "NOT-FOR-PRICING" in p["use_restriction"]
    texts = [p["disclaimer"], p["class_adjacency"], p["self_report_badge"],
             p["site_only_note"], *p["class_meaning"].values()]
    for t in texts:
        assert scan_text(t) == [], t


def _cases() -> list[dict]:
    cases = []
    tri = (None, True, False)
    for st, year, floors, src, soft, flags, retro in itertools.product(
        STRUCTURE_TYPES, (None, 1974, 1975, 1990, 2006, 2012, 2018, 2025), (None, 1, 4, 7, 16, 60),
        ("self_reported", "surveyed", "official"), tri, tri, (None, False, True),
    ):
        card = dict(structure_type=st, year_built=year, n_floors=floors, soft_story=soft,
                    irregularity=flags, pounding_risk=flags, prior_unrepaired_damage=flags,
                    amnesty_imar_barisi=flags, retrofit=retro, source=src)
        if retro:
            card["retrofit_year"] = 2024
        cases.append(dict(site_risk=(len(cases) * 7) % 101, card=card, year=YEAR))
    # Geçersiz karneler: aynı alan hataları
    for bad in (dict(year_built=2300), dict(n_floors=0), dict(source="x"),
                dict(structure_type="adobe"), dict(retrofit=False, retrofit_year=2010),
                dict(year_built=2300, retrofit=True, retrofit_year=2024)):
        cases.append(dict(site_risk=50, card={"year_built": 2000, **bad}, year=YEAR))
    return cases


def _python(case: dict) -> dict:
    from eyequake.analysis.fragility import CardValidationError

    try:
        o = apply_building_overlay(case["site_risk"], BuildingCard(**case["card"]), case["year"])
    except CardValidationError as exc:
        return {"errors": [f for f, _ in exc.errors]}
    a = o.assessment
    return {
        "parcel_risk": o.parcel_risk, "resolution": o.resolution,
        "multiplier_effective": o.multiplier_effective, "downward_capped": o.downward_capped,
        "v": a.vulnerability_index, "cls": a.vulnerability_class, "m_raw": a.multiplier_raw,
        "kappa": a.kappa, "warnings": list(a.plausibility_warnings),
        "imputed": list(a.imputed_fields),
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node yok — JS paritesi atlandı")
def test_js_matches_python_on_card_grid(tmp_path):
    cases = _cases()
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "params.json").write_text(json.dumps(export_params()), encoding="utf-8")
    (tmp_path / "run.js").write_text(_NODE_RUNNER, encoding="utf-8")
    res = subprocess.run(
        ["node", str(tmp_path / "run.js"), str(ROOT / "web" / "fragility.js"),
         str(tmp_path / "params.json"), str(tmp_path / "cases.json")],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    got = json.loads(res.stdout)
    assert got["selfcheck"] == []
    mismatches = [
        (c["card"], c["site_risk"], js, py)
        for c, js, py in zip(cases, got["out"], map(_python, cases))
        if js != py
    ]
    assert not mismatches, f"{len(mismatches)}/{len(cases)} sapma, ilk: {mismatches[0]}"
