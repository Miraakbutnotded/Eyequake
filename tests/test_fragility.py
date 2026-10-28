"""fragility.py birim testleri — BIS Faz-1 bina karnesi (tasarım §6.3 test listesi).

Çalıştır: ./.venv/bin/python -m pytest tests/test_fragility.py -v
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from eyequake.analysis.fragility import (
    DISCLAIMER,
    SELF_REPORT_BADGE,
    STRUCTURE_TYPES,
    BuildingCard,
    CardValidationError,
    apply_building_overlay,
    assess_building,
    era_from_year,
)
from eyequake.integrity import scan_text

YEAR = 2026
ROOT = Path(__file__).resolve().parents[1]


def _full(**kw) -> BuildingCard:
    """Tüm alanları dolu karne (tamlık = 1.0); kw ile alan ezilir."""
    base = dict(
        structure_type="rc_frame", year_built=2010, n_floors=4, soft_story=False,
        irregularity=False, pounding_risk=False, prior_unrepaired_damage=False,
        amnesty_imar_barisi=False, retrofit=False, source="surveyed",
    )
    base.update(kw)
    return BuildingCard(**base)


@pytest.mark.parametrize("year,era", [
    (1974, "pre1975"), (1975, "1975_1997"), (1997, "1975_1997"), (1998, "1998_2006"),
    (2006, "1998_2006"), (2007, "2007_2017"), (2017, "2007_2017"), (2018, "post2018"),
])
def test_era_from_year_boundaries(year, era):
    assert era_from_year(year) == era


def test_older_weaker_type_more_vulnerable():
    old = assess_building(_full(structure_type="rc_frame", year_built=1970), YEAR)
    new = assess_building(_full(structure_type="rc_shearwall", year_built=2020), YEAR)
    assert old.vulnerability_index > new.vulnerability_index


@pytest.mark.parametrize("flag", [
    "soft_story", "irregularity", "pounding_risk", "prior_unrepaired_damage",
    "amnesty_imar_barisi",
])
def test_modifiers_strictly_increase_v(flag):
    base = assess_building(_full(), YEAR).vulnerability_index
    assert assess_building(_full(**{flag: True}), YEAR).vulnerability_index > base


def test_soft_story_and_irregularity_stack():
    one = assess_building(_full(soft_story=True), YEAR).vulnerability_index
    two = assess_building(_full(soft_story=True, irregularity=True), YEAR).vulnerability_index
    assert two > one


def test_retrofit_credit_depends_on_trust():
    def v(source, retrofit):
        return assess_building(
            _full(source=source, retrofit=retrofit, retrofit_year=2020 if retrofit else None),
            YEAR,
        ).vulnerability_index

    self_credit = v("self_reported", False) - v("self_reported", True)
    surveyed_credit = v("surveyed", False) - v("surveyed", True)
    assert surveyed_credit > self_credit > 0
    assert self_credit == pytest.approx(0.05, abs=1e-3)


def test_masonry_over_three_floors_penalised():
    low = assess_building(_full(structure_type="confined_masonry", n_floors=3), YEAR)
    high = assess_building(_full(structure_type="confined_masonry", n_floors=4), YEAR)
    assert high.vulnerability_index == pytest.approx(low.vulnerability_index + 0.10)


def test_bounds_over_input_grid():
    for st, year, floors, src, soft, retro in itertools.product(
        STRUCTURE_TYPES, (1950, 1990, 2003, 2012, 2022), (1, 5, 30, 60),
        ("self_reported", "surveyed", "official"), (None, True), (False, True),
    ):
        card = BuildingCard(
            structure_type=st, year_built=year, n_floors=floors, soft_story=soft,
            irregularity=soft, prior_unrepaired_damage=soft, amnesty_imar_barisi=soft,
            retrofit=retro, source=src,
        )
        a = assess_building(card, YEAR)
        assert 0.05 <= a.vulnerability_index <= 0.98
        assert 0.65 <= a.multiplier_raw <= 1.80
        assert 0.0 <= a.kappa <= 1.0
        for site_risk in (0, 37, 100):
            o = apply_building_overlay(site_risk, card, YEAR)
            assert 0 <= o.parcel_risk <= 100
            lo, hi = min(1.0, a.multiplier_raw), max(1.0, a.multiplier_raw)
            assert lo - 1e-9 <= o.multiplier_effective <= hi + 1e-9 or o.downward_capped


def test_no_card_is_site_only():
    o = apply_building_overlay(61, None)
    assert (o.parcel_risk, o.applied, o.resolution, o.reason) == (61, False, "site_only", "no_card")
    assert o.multiplier_effective == 1.0


def test_unknown_everything_regresses_to_site_risk():
    o = apply_building_overlay(61, BuildingCard(), YEAR)
    assert o.parcel_risk == 61
    assert o.resolution == "site_only" and o.reason == "low_confidence"
    assert set(o.assessment.imputed_fields) == {"structure_type", "year_built"}


def test_partial_card_shrinks_between_one_and_raw():
    card = BuildingCard(structure_type="rc_frame", year_built=1985, n_floors=6,
                        soft_story=True, source="self_reported")
    o = apply_building_overlay(50, card, YEAR)
    a = o.assessment
    assert o.resolution == "building_partial"
    assert a.multiplier_raw > 1.0
    assert 1.0 < o.multiplier_effective < a.multiplier_raw


def test_self_report_cannot_cut_more_than_ten_percent():
    favorable = dict(structure_type="steel", year_built=2022, n_floors=2,
                     retrofit=True, retrofit_year=2024)
    official = apply_building_overlay(80, _full(source="official", **favorable), YEAR)
    assert official.multiplier_effective < 0.90
    # Özellik: hiçbir beyan kombinasyonu skoru %10'dan fazla düşüremez.
    for st, year, retro in itertools.product(STRUCTURE_TYPES, (1950, 2003, 2022), (False, True)):
        card = _full(source="self_reported", structure_type=st, year_built=year,
                     n_floors=2, retrofit=retro, retrofit_year=2024 if retro else None)
        assert apply_building_overlay(80, card, YEAR).multiplier_effective >= 0.90


def test_worse_self_report_applied():
    card = _full(source="self_reported", year_built=1980, n_floors=7, soft_story=True)
    o = apply_building_overlay(50, card, YEAR)
    assert o.applied and o.parcel_risk > 50
    assert SELF_REPORT_BADGE in o.notes


@pytest.mark.parametrize("kw,bad_field", [
    (dict(year_built=2300), "year_built"),
    (dict(year_built=1800), "year_built"),
    (dict(n_floors=0), "n_floors"),
    (dict(n_floors=61), "n_floors"),
    (dict(retrofit=True, retrofit_year=2000), "retrofit_year"),   # < year_built
    (dict(retrofit=False, retrofit_year=2015), "retrofit_year"),
    (dict(structure_type="adobe"), "structure_type"),
    (dict(source="neighbour"), "source"),
])
def test_validation_rejects_with_field(kw, bad_field):
    with pytest.raises(CardValidationError) as exc:
        assess_building(_full(**kw), YEAR)
    assert bad_field in [f for f, _ in exc.value.errors]


def test_validation_collects_all_errors():
    with pytest.raises(CardValidationError) as exc:
        assess_building(_full(year_built=2300, n_floors=0, source="x"), YEAR)
    assert {f for f, _ in exc.value.errors} == {"year_built", "n_floors", "source"}


def test_golden_archetypes():
    weak = apply_building_overlay(
        60, _full(year_built=1995, n_floors=7, soft_story=True), YEAR)
    strong = apply_building_overlay(
        60, _full(structure_type="rc_shearwall", year_built=2020, n_floors=4), YEAR)
    assert weak.assessment.vulnerability_class in ("A", "B")
    assert weak.multiplier_effective > 1.0 and weak.resolution == "building_full"
    assert strong.assessment.vulnerability_class in ("E", "F")
    assert strong.multiplier_effective < 1.0
    assert apply_building_overlay(60, BuildingCard(), YEAR).resolution == "site_only"


def test_implausible_combo_flagged_not_rejected():
    a = assess_building(_full(structure_type="unreinforced_masonry", n_floors=15), YEAR)
    assert a.plausibility_warnings


def test_deterministic():
    card = _full(year_built=1988, soft_story=True, source="self_reported")
    assert apply_building_overlay(55, card, YEAR) == apply_building_overlay(55, card, YEAR)


def test_outward_copy_passes_integrity_linter():
    assert scan_text(DISCLAIMER) == []
    for note in apply_building_overlay(50, _full(source="self_reported"), YEAR).notes:
        assert scan_text(note) == [], note


def test_building_collapse_claims_are_banned():
    assert scan_text("Bu bina çökecek.")
    assert scan_text("Model hasar tahmini yapar.")
    assert scan_text("Hasar/çökme tahmini değildir.") == []


def test_model_card_keeps_pricing_gate_closed():
    text = (ROOT / "reports" / "FRAGILITY_MODEL_CARD.md").read_text(encoding="utf-8")
    assert "sonuç-doğrulaması yapılmadı" in text
    assert "fiyatlama kullanımı kapalı" in text.lower()
