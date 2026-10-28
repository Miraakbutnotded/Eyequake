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
    if year < 1975:
        return "pre1975"
    if year < 1998:
        return "1975_1997"
    if year < 2007:
        return "1998_2006"
    if year < 2018:
        return "2007_2017"
    return "post2018"


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
        lo = card.year_built if card.year_built is not None else _YEAR_MIN
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
            v += 0.10 if card.n_floors > 3 else 0.0
        elif card.structure_type in _FLOOR_PENALTY_TYPES:
            v += min(0.18, max(0.0, 0.018 * (card.n_floors - 3)))
    for name, delta in _MODIFIERS.items():
        if getattr(card, name):
            v += delta
    if card.retrofit:
        v -= _RETROFIT_CREDIT * _RETROFIT_TRUST[card.source]
    v = round(min(_V_MAX, max(_V_MIN, v)), 3)

    m_raw = round(min(1.80, max(0.65, 0.65 + 1.15 * v)), 2)
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
    if card.structure_type in _MASONRY and card.n_floors is not None and card.n_floors > 5:
        warnings.append("yığma yapı için olağandışı kat sayısı")
    if era == "pre1975" and card.n_floors is not None and card.n_floors >= 15:
        warnings.append("1975 öncesi için olağandışı yüksek bina")

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
