"""Bina kırılganlık katmanı (BIS Faz-1) — kullanıcı-girişli bina karnesi.

Bina karnesi (yapı tipi, yapım yılı, kat, yumuşak kat, düzensizlik, güçlendirme …)
→ göreli kırılganlık indeksi V → EMS-98 tarzı sınıf (A..F) → sınırlı çarpan. Çarpan
sorgu anında hazard × saha yüzeyinin (`site.site_adjusted_risk`) üstüne bindirilir.

Saf, test edilebilir fonksiyonlar (site.py'nin sınırlı-bin desenini izler):
  - era_from_year          : yapım yılı → yönetmelik dönemi
  - validate_card          : karne doğrulaması (alan-bazlı hata listesi)
  - assess_building        : karne → V, sınıf, M_raw, tamlık, güven (κ)
  - apply_building_overlay : site_risk × M_eff → parsel-risk (+ çözünürlük bayrağı)

Bilimsel sınır: uzman-türetilmiş taksonomi (EMS-98 × TBDY dönemi × yapı tipi),
eğitilmiş model DEĞİL — Türkiye için etiketli kırılganlık yer-gerçeği yok. 2023
hasar-tespit verisine karşı sonuç-doğrulaması yapılmadı; fiyatlama kullanımı kapalı
(bkz. reports/FRAGILITY_MODEL_CARD.md). Hasar/çökme tahmini değildir.

Kaynak tasarım: docs/plans/2026-06-28-end-to-end-production-system-design.md §6.3.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

FRAGILITY_MODEL_VERSION = "frag-1.0.0"

DISCLAIMER = "Göreli yapısal kırılganlık göstergesi — hasar/çökme tahmini DEĞİLDİR."

# Her render'da sınıfın yanında zorunlu (UI/API kopya kontratı, §6.3).
CLASS_ADJACENCY = (
    "Göreli, tanımlayıcı yapısal kırılganlık sınıfı (EMS-98 tarzı); mühendislik hasar "
    "tespiti / RBTE yerine geçmez, çökme/hasar tahmini değildir."
)
SELF_REPORT_BADGE = "Beyana dayalı — doğrulanmamış."
SITE_ONLY_NOTE = "Bina verisi yok → yalnız zemin × hazard skoru."

STRUCTURE_TYPES = (
    "rc_frame",
    "rc_shearwall",
    "confined_masonry",
    "unreinforced_masonry",
    "precast_industrial",
    "steel",
    "timber_other",
    "unknown",
)
SOURCES = ("self_reported", "surveyed", "official")
ERAS = ("pre1975", "1975_1997", "1998_2006", "2007_2017", "post2018")
# Dönem kırılım yılları (ERAS[i+1] bu yıldan itibaren): 1975, 1998, 2007, 2018 yönetmelikleri.
_ERA_BREAKS = (1975, 1998, 2007, 2018)

# V_base[yapı tipi][dönem] ∈ [0,1], 1 = en kırılgan. Dönemler ERAS sırasıyla.
_BASE = {
    "unreinforced_masonry": (0.85, 0.80, 0.72, 0.65, 0.55),
    "confined_masonry": (0.70, 0.62, 0.52, 0.45, 0.38),
    "rc_frame": (0.78, 0.70, 0.55, 0.42, 0.30),
    "rc_shearwall": (0.60, 0.52, 0.40, 0.30, 0.20),
    "precast_industrial": (0.72, 0.65, 0.55, 0.45, 0.35),
    "steel": (0.45, 0.40, 0.33, 0.27, 0.20),
    "timber_other": (0.60, 0.55, 0.50, 0.45, 0.40),
}
_MASONRY = ("unreinforced_masonry", "confined_masonry")
# Kat cezası rijit olmayan tiplere; perdeli betonarme ve çelik muaf. Yığma ayrı dal
# (>3 kat → +0.10). Bilinmeyen tip ihtiyatlı tarafta tutulur → cezaya dahil.
_FLOOR_PENALTY_TYPES = ("rc_frame", "precast_industrial", "timber_other", "unknown")
_FLOOR_BASE, _FLOOR_STEP, _FLOOR_CAP = 3, 0.018, 0.18
_MASONRY_FLOOR_LIMIT, _MASONRY_PENALTY = 3, 0.10

_MODIFIERS = {
    "soft_story": 0.15,
    "irregularity": 0.08,
    "pounding_risk": 0.05,
    "prior_unrepaired_damage": 0.12,
    "amnesty_imar_barisi": 0.10,
}
_RETROFIT_CREDIT = 0.25
_RETROFIT_TRUST = {"self_reported": 0.2, "surveyed": 1.0, "official": 1.0}
_SOURCE_TRUST = {"self_reported": 0.5, "surveyed": 0.85, "official": 1.0}

# Tamlık ağırlıkları: yapı tipi ve dönem 3×, kat ve yumuşak kat 2×, kalanı 1×.
_COMPLETENESS_WEIGHTS = {
    "structure_type": 3,
    "year_built": 3,
    "n_floors": 2,
    "soft_story": 2,
    "irregularity": 1,
    "pounding_risk": 1,
    "prior_unrepaired_damage": 1,
    "amnesty_imar_barisi": 1,
    "retrofit": 1,
}

_V_MIN, _V_MAX = 0.05, 0.98
# M_raw = clip(_M_INTERCEPT + _M_SLOPE · V, _M_MIN, _M_MAX)
_M_INTERCEPT, _M_SLOPE, _M_MIN, _M_MAX = 0.65, 1.15, 0.65, 1.80
_CLASS_BINS = ((0.80, "A"), (0.65, "B"), (0.50, "C"), (0.35, "D"), (0.20, "E"))
CLASS_MEANING = {
    "A": "çok yüksek göreli kırılganlık",
    "B": "yüksek",
    "C": "orta-yüksek",
    "D": "orta",
    "E": "düşük",
    "F": "çok düşük",
}
_KAPPA_FULL, _KAPPA_PARTIAL = 0.7, 0.3
_SELF_REPORT_FLOOR = 0.90   # beyan skoru en fazla %10 düşürebilir (anti-gaming)
_YEAR_MIN, _FLOORS_MIN, _FLOORS_MAX = 1900, 1, 60
# Olağandışı kombinasyon uyarıları (reddedilmez, işaretlenir).
_MASONRY_TALL_FLOORS, _PRE1975_TALL_FLOORS = 5, 15
_WARN_MASONRY_TALL = "yığma yapı için olağandışı kat sayısı"
_WARN_PRE1975_TALL = "1975 öncesi için olağandışı yüksek bina"


@dataclass(frozen=True)
class BuildingCard:
    """Kullanıcı-girişli bina karnesi. None = bilinmiyor (tamlığı düşürür)."""

    structure_type: str = "unknown"
    year_built: int | None = None
    n_floors: int | None = None
    soft_story: bool | None = None
    irregularity: bool | None = None
    pounding_risk: bool | None = None
    prior_unrepaired_damage: bool | None = None
    amnesty_imar_barisi: bool | None = None
    retrofit: bool | None = None
    retrofit_year: int | None = None
    source: str = "self_reported"
    evidence_ref: str | None = None


class CardValidationError(ValueError):
    """Geçersiz karne. ``errors`` = [(alan, mesaj), ...] — hiçbir şey hesaplanmaz."""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{f}: {m}" for f, m in errors))


@dataclass(frozen=True)
class BuildingAssessment:
    vulnerability_index: float
    vulnerability_class: str
    multiplier_raw: float
    completeness: float
    source_trust: float
    kappa: float
    resolution: str
    trust_flag: str
    imputed_fields: tuple[str, ...] = ()
    plausibility_warnings: tuple[str, ...] = ()
    fragility_model_version: str = FRAGILITY_MODEL_VERSION


@dataclass(frozen=True)
class BuildingOverlay:
    parcel_risk: int
    site_risk: int
    applied: bool
    resolution: str
    multiplier_effective: float
    downward_capped: bool = False
    reason: str | None = None
    assessment: BuildingAssessment | None = None
    disclaimer: str = DISCLAIMER
    notes: tuple[str, ...] = field(default=())


def era_from_year(year: int) -> str:
    """Yapım yılı → TR deprem yönetmeliği dönemi (1975, 1998, 2007, 2018 kırılımları)."""
    return ERAS[sum(year >= b for b in _ERA_BREAKS)]


def validate_card(card: BuildingCard, current_year: int | None = None) -> None:
    """Karneyi doğrula; tüm alan hatalarını toplayıp CardValidationError fırlat."""
    now = current_year or _dt.date.today().year
    errors: list[tuple[str, str]] = []
    if card.structure_type not in STRUCTURE_TYPES:
        errors.append(("structure_type", f"geçersiz değer: {card.structure_type!r}"))
    if card.source not in SOURCES:
        errors.append(("source", f"geçersiz değer: {card.source!r}"))
    if card.year_built is not None and not _YEAR_MIN <= card.year_built <= now:
        errors.append(("year_built", f"[{_YEAR_MIN}, {now}] aralığında olmalı"))
    if card.n_floors is not None and not _FLOORS_MIN <= card.n_floors <= _FLOORS_MAX:
        errors.append(("n_floors", f"[{_FLOORS_MIN}, {_FLOORS_MAX}] aralığında olmalı"))
    if card.retrofit_year is not None:
        if not card.retrofit:
            errors.append(("retrofit_year", "retrofit=True olmadan verilemez"))
        year_ok = card.year_built is not None and _YEAR_MIN <= card.year_built <= now
        lo = card.year_built if year_ok else _YEAR_MIN
        if not lo <= card.retrofit_year <= now:
            errors.append(("retrofit_year", f"[{lo}, {now}] aralığında olmalı"))
    if errors:
        raise CardValidationError(errors)


def _base_vulnerability(structure_type: str, era: str | None) -> float:
    """V_base; bilinmeyen tip/dönem → dönem/tip kohort ortalaması (iyimser değil)."""
    types = [structure_type] if structure_type in _BASE else list(_BASE)
    eras = [ERAS.index(era)] if era is not None else range(len(ERAS))
    vals = [_BASE[t][i] for t in types for i in eras]
    return sum(vals) / len(vals)


def _completeness(card: BuildingCard) -> float:
    provided = {
        name: getattr(card, name) is not None for name in _COMPLETENESS_WEIGHTS
    }
    provided["structure_type"] = card.structure_type != "unknown"
    total = sum(_COMPLETENESS_WEIGHTS.values())
    got = sum(w for name, w in _COMPLETENESS_WEIGHTS.items() if provided[name])
    return got / total


def vulnerability_class(v: float) -> str:
    for lower, cls in _CLASS_BINS:
        if v >= lower:
            return cls
    return "F"


def assess_building(card: BuildingCard, current_year: int | None = None) -> BuildingAssessment:
    """Karne → kırılganlık değerlendirmesi. Önce doğrular (CardValidationError)."""
    validate_card(card, current_year)
    era = era_from_year(card.year_built) if card.year_built is not None else None
    imputed = []
    if card.structure_type == "unknown":
        imputed.append("structure_type")
    if era is None:
        imputed.append("year_built")

    v = _base_vulnerability(card.structure_type, era)
    if card.n_floors is not None:
        if card.structure_type in _MASONRY:
            v += _MASONRY_PENALTY if card.n_floors > _MASONRY_FLOOR_LIMIT else 0.0
        elif card.structure_type in _FLOOR_PENALTY_TYPES:
            v += min(_FLOOR_CAP, max(0.0, _FLOOR_STEP * (card.n_floors - _FLOOR_BASE)))
    for name, delta in _MODIFIERS.items():
        if getattr(card, name):
            v += delta
    if card.retrofit:
        v -= _RETROFIT_CREDIT * _RETROFIT_TRUST[card.source]
    v = round(min(_V_MAX, max(_V_MIN, v)), 3)

    m_raw = round(min(_M_MAX, max(_M_MIN, _M_INTERCEPT + _M_SLOPE * v)), 2)
    completeness = round(_completeness(card), 2)
    source_trust = _SOURCE_TRUST[card.source]
    kappa = round(min(1.0, max(0.0, completeness * source_trust)), 2)
    if kappa >= _KAPPA_FULL:
        resolution = "building_full"
    elif kappa >= _KAPPA_PARTIAL:
        resolution = "building_partial"
    else:
        resolution = "site_only"

    warnings = []
    floors = card.n_floors
    if card.structure_type in _MASONRY and floors is not None and floors > _MASONRY_TALL_FLOORS:
        warnings.append(_WARN_MASONRY_TALL)
    if era == ERAS[0] and floors is not None and floors >= _PRE1975_TALL_FLOORS:
        warnings.append(_WARN_PRE1975_TALL)

    return BuildingAssessment(
        vulnerability_index=v,
        vulnerability_class=vulnerability_class(v),
        multiplier_raw=m_raw,
        completeness=completeness,
        source_trust=source_trust,
        kappa=kappa,
        resolution=resolution,
        trust_flag=card.source,
        imputed_fields=tuple(imputed),
        plausibility_warnings=tuple(warnings),
    )


def apply_building_overlay(site_risk: int, card: BuildingCard | None,
                           current_year: int | None = None) -> BuildingOverlay:
    """site_risk × M_eff → parsel-risk. site_risk'in parsel-risk'e dönüştüğü TEK yer.

    Kredibilite büzülmesi: M_eff = 1 + (M_raw − 1)·κ; κ < 0.3 → M_eff = 1 (site_only).
    Beyan (self_reported) skoru en fazla %10 düşürebilir; kötüleştiren beyan tam uygulanır.
    """
    if card is None:
        return BuildingOverlay(
            parcel_risk=site_risk, site_risk=site_risk, applied=False,
            resolution="site_only", multiplier_effective=1.0, reason="no_card",
            notes=(SITE_ONLY_NOTE,),
        )
    a = assess_building(card, current_year)
    if a.resolution == "site_only":
        return BuildingOverlay(
            parcel_risk=site_risk, site_risk=site_risk, applied=False,
            resolution="site_only", multiplier_effective=1.0, reason="low_confidence",
            assessment=a, notes=(SITE_ONLY_NOTE,),
        )
    m_eff = 1.0 + (a.multiplier_raw - 1.0) * a.kappa
    capped = False
    if a.trust_flag == "self_reported" and m_eff < _SELF_REPORT_FLOOR:
        m_eff, capped = _SELF_REPORT_FLOOR, True
    m_eff = round(m_eff, 3)
    notes = [CLASS_ADJACENCY]
    if a.trust_flag == "self_reported":
        notes.append(SELF_REPORT_BADGE)
    return BuildingOverlay(
        parcel_risk=int(max(0, min(100, round(site_risk * m_eff)))),
        site_risk=site_risk, applied=True, resolution=a.resolution,
        multiplier_effective=m_eff, downward_capped=capped, assessment=a,
        notes=tuple(notes),
    )


# --- Web export (tek doğruluk kaynağı) -------------------------------------------
# Statik web (backend yok) aynı formülü JS'te koşar; parametreler ve metinler buradan
# web/data/fragility_params.json'a yazılır (scripts/13_export_fragility_params.py).
# Altın vektörler: JS yüklemede bunları yeniden hesaplar; sapma → form kapanır.

_GOLDEN_CARDS = (
    (60, dict(structure_type="rc_frame", year_built=1995, n_floors=7, soft_story=True,
              irregularity=False, pounding_risk=False, prior_unrepaired_damage=False,
              amnesty_imar_barisi=False, retrofit=False, source="surveyed")),
    (60, dict(structure_type="rc_shearwall", year_built=2020, n_floors=4, soft_story=False,
              irregularity=False, pounding_risk=False, prior_unrepaired_damage=False,
              amnesty_imar_barisi=False, retrofit=False, source="surveyed")),
    (50, dict(structure_type="rc_frame", year_built=1985, n_floors=6, soft_story=True)),
    (80, dict(structure_type="steel", year_built=2022, n_floors=2, retrofit=True,
              retrofit_year=2024)),
    (45, dict(structure_type="unreinforced_masonry", year_built=1960, n_floors=4,
              amnesty_imar_barisi=True, prior_unrepaired_damage=True, soft_story=False)),
    (70, dict(structure_type="unknown", n_floors=12, soft_story=True, irregularity=True)),
    (61, dict()),
    (97, dict(structure_type="precast_industrial", year_built=1970, n_floors=60,
              soft_story=True, irregularity=True, pounding_risk=True,
              prior_unrepaired_damage=True, amnesty_imar_barisi=True, retrofit=False)),
)


_GOLDEN_YEAR = 2026   # altın vektörler sabit yılla hesaplanır (deterministik export)


def _golden_vectors() -> list[dict]:
    out = []
    for site_risk, kw in _GOLDEN_CARDS:
        o = apply_building_overlay(site_risk, BuildingCard(**kw), current_year=_GOLDEN_YEAR)
        a = o.assessment
        out.append({
            "site_risk": site_risk,
            "card": kw,
            "expected": {
                "parcel_risk": o.parcel_risk,
                "applied": o.applied,
                "resolution": o.resolution,
                "multiplier_effective": o.multiplier_effective,
                "downward_capped": o.downward_capped,
                "vulnerability_index": a.vulnerability_index,
                "vulnerability_class": a.vulnerability_class,
                "multiplier_raw": a.multiplier_raw,
                "completeness": a.completeness,
                "kappa": a.kappa,
            },
        })
    return out


def export_params() -> dict:
    """Web'in JS uygulaması için tüm parametre + kopya kontratı (deterministik)."""
    return {
        "fragility_model_version": FRAGILITY_MODEL_VERSION,
        "source": "Uzman-türetilmiş taksonomi (EMS-98 × TBDY dönemi × yapı tipi), "
                  "sonuç-doğrulaması yok. Bkz. reports/FRAGILITY_MODEL_CARD.md",
        "use_restriction": "NOT-FOR-PRICING — göreli triage göstergesi, fiyatlama kullanımı kapalı.",
        "disclaimer": DISCLAIMER,
        "class_adjacency": CLASS_ADJACENCY,
        "self_report_badge": SELF_REPORT_BADGE,
        "site_only_note": SITE_ONLY_NOTE,
        "class_meaning": CLASS_MEANING,
        "structure_types": list(STRUCTURE_TYPES),
        "sources": list(SOURCES),
        "eras": list(ERAS),
        "era_breaks": list(_ERA_BREAKS),
        "base": {t: list(v) for t, v in _BASE.items()},
        "masonry": list(_MASONRY),
        "floor_penalty_types": list(_FLOOR_PENALTY_TYPES),
        "floor": {"base": _FLOOR_BASE, "step": _FLOOR_STEP, "cap": _FLOOR_CAP},
        "masonry_floor": {"limit": _MASONRY_FLOOR_LIMIT, "penalty": _MASONRY_PENALTY},
        "modifiers": [[k, v] for k, v in _MODIFIERS.items()],
        "retrofit_credit": _RETROFIT_CREDIT,
        "retrofit_trust": _RETROFIT_TRUST,
        "source_trust": _SOURCE_TRUST,
        "completeness_weights": [[k, w] for k, w in _COMPLETENESS_WEIGHTS.items()],
        "v_bounds": [_V_MIN, _V_MAX],
        "multiplier": {"intercept": _M_INTERCEPT, "slope": _M_SLOPE,
                       "min": _M_MIN, "max": _M_MAX},
        "class_bins": [[lo, c] for lo, c in _CLASS_BINS],
        "class_fallback": "F",
        "kappa_thresholds": {"full": _KAPPA_FULL, "partial": _KAPPA_PARTIAL},
        "self_report_floor": _SELF_REPORT_FLOOR,
        "validation": {"year_min": _YEAR_MIN, "floors_min": _FLOORS_MIN,
                       "floors_max": _FLOORS_MAX},
        "plausibility": {"masonry_tall_floors": _MASONRY_TALL_FLOORS,
                         "pre1975_tall_floors": _PRE1975_TALL_FLOORS,
                         "warn_masonry_tall": _WARN_MASONRY_TALL,
                         "warn_pre1975_tall": _WARN_PRE1975_TALL},
        "golden_year": _GOLDEN_YEAR,
        "golden": _golden_vectors(),
    }
