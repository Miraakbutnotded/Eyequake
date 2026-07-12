"""Run and validate the EyeQuake reproducibility pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.analysis.seismology import (  # noqa: E402
    b_value_aki,
    magnitude_of_completeness,
)
from eyequake.config import TURKEY  # noqa: E402
from eyequake.forecast.spatial_skill import evaluate_surface_skill  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = Path("data/processed")
WEB_DATA_DIR = Path("web/data")


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce or validate expected outputs."""


@dataclass(frozen=True)
class PipelineOptions:
    source: str = "afad"
    skip_fetch: bool = False
    skip_site_layer: bool = False
    export_web: bool = False
    check: bool = False
    check_surface: bool = False


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> PipelineOptions:
    parser = argparse.ArgumentParser(
        description="Run EyeQuake data/model/web export pipeline and validation gates."
    )
    parser.add_argument("--source", choices=["afad"], default="afad")
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing processed catalog.")
    parser.add_argument("--skip-site-layer", action="store_true", help="Skip elevation-backed site layer.")
    parser.add_argument("--export-web", action="store_true", help="Regenerate web/data outputs.")
    parser.add_argument("--check", action="store_true", help="Only validate existing outputs.")
    parser.add_argument(
        "--check-surface",
        action="store_true",
        help="Validate only the committed web/data surface contract (no catalog/network; CI gate).",
    )
    args = parser.parse_args(argv)
    return PipelineOptions(
        source=args.source,
        skip_fetch=args.skip_fetch,
        skip_site_layer=args.skip_site_layer,
        export_web=args.export_web,
        check=args.check,
        check_surface=args.check_surface,
    )


def build_steps(options: PipelineOptions) -> list[Step]:
    if options.source != "afad":
        raise PipelineError(f"Unsupported source: {options.source}")
    if options.check or options.check_surface:
        return []

    py = sys.executable
    steps: list[Step] = []
    if not options.skip_fetch:
        steps.append(Step("fetch-afad", (py, "scripts/07_fetch_afad.py")))
    if options.export_web:
        steps.append(Step("export-web-data", (py, "scripts/06_export_web_data.py")))
        if not options.skip_site_layer:
            steps.append(Step("site-layer", (py, "scripts/10_site_layer.py")))
        steps.append(Step("export-etas-params", (py, "scripts/11_export_etas_params.py")))
    steps.extend(
        [
            Step("ruff", (py, "-m", "ruff", "check", ".")),
            Step(
                "pytest",
                (py, "-m", "pytest", "--cov=eyequake", "--cov-report=term-missing"),
            ),
        ]
    )
    return steps


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Expected JSON object: {path}")
    return payload


def require_file(path: Path) -> None:
    if not path.exists():
        raise PipelineError(f"Missing required file: {path}")
    if path.stat().st_size == 0:
        raise PipelineError(f"Empty required file: {path}")


def require_keys(label: str, payload: dict[str, Any], keys: set[str]) -> None:
    missing = sorted(keys - payload.keys())
    if missing:
        raise PipelineError(f"{label} missing required keys: {', '.join(missing)}")


def assert_use_restriction(meta: dict[str, Any]) -> None:
    """E&O / liability guard: meta must declare the NOT-FOR-PRICING boundary.

    The score is a RELATIVE descriptive percentile index, not calibrated to
    absolute loss. Selling it to insurers is fine; letting it be used as a direct
    premium-rating cat model is an E&O exposure. This encodes that boundary as a
    machine-checked contract: the restriction text must mention rating/pricing AND
    explicitly negate it (must say it is NOT a rating input). A present-but-
    affirmative value (e.g. "fiyatlama için hazır") is rejected just like absence.

    Negation is Turkish-only ("değil"/"degil"). An English "not" substring needle
    was REMOVED: it is a false-negative hole — "fiyatlama notu" and "rating note:
    ready for pricing" both embed "not" and would wrongly PASS. The copy is Turkish
    and "değil" is word-ish (not a substring of değiş/değer/değişiklik), so it has
    no analogous hole.
    """
    # Turkish-robust lowercase: str.lower() maps "İ" to "i"+combining-dot, which
    # would break a plain "değil" substring test; pre-map the dotted capital I.
    text = str(meta.get("use_restriction", "")).replace("İ", "i").lower()
    has_rating_marker = "rating" in text or "fiyatlama" in text
    has_negation = "değil" in text or "degil" in text
    if not (has_rating_marker and has_negation):
        raise PipelineError(
            "meta use_restriction must state the NOT-FOR-PRICING boundary "
            "(premium rating / fiyatlama için doğrudan girdi DEĞİLDİR)."
        )


def first_feature(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("type") != "FeatureCollection":
        raise PipelineError(f"{path} must be a FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise PipelineError(f"{path} must contain at least one feature")
    feature = features[0]
    if not isinstance(feature, dict):
        raise PipelineError(f"{path} has invalid feature payload")
    return feature


def catalog_event_count(path: Path) -> int:
    require_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PipelineError(f"Catalog has no header: {path}")
        required = {"time", "latitude", "longitude", "mag"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise PipelineError(f"Catalog missing columns: {', '.join(missing)}")
        return sum(1 for _ in reader)


def validate_outputs(root: Path = ROOT, require_site_layer: bool = True) -> dict[str, Any]:
    processed = root / PROCESSED_DIR
    web_data = root / WEB_DATA_DIR

    catalog_path = processed / "turkey_catalog.csv"
    summary_path = processed / "turkey_summary.json"
    meta_path = web_data / "meta.json"
    hazard_path = web_data / "hazard_cells.geojson"
    big_events_path = web_data / "big_events.geojson"
    etas_path = web_data / "etas_params.json"

    catalog_events = catalog_event_count(catalog_path)
    if catalog_events <= 0:
        raise PipelineError("Catalog must contain at least one event")

    summary = load_json(summary_path)
    require_keys("summary", summary, {"source", "n_events", "date_min", "date_max"})
    if summary["source"] != "AFAD":
        raise PipelineError(f"summary source must be AFAD, got {summary['source']!r}")

    meta = load_json(meta_path)
    require_keys(
        "meta",
        meta,
        {"source", "n_cells", "n_big_events", "disclaimer", "mc", "use_restriction"},
    )
    assert_use_restriction(meta)
    if meta["source"] != "AFAD":
        raise PipelineError(f"meta source must be AFAD, got {meta['source']!r}")
    if "tahmini" not in str(meta["disclaimer"]).lower():
        raise PipelineError("meta disclaimer must state forecast/prediction limits")

    hazard_feature = first_feature(hazard_path)
    props = hazard_feature.get("properties")
    if not isinstance(props, dict):
        raise PipelineError("hazard feature missing properties")
    required_hazard = {
        "risk_model",
        "risk_index",
        "annual_rate",
        "recent_rate",
        "recency_days",
        "energy_index",
        "shallow_fraction",
    }
    if require_site_layer:
        required_hazard |= {"site_risk", "site_class", "amp", "liq_class"}
    require_keys("hazard feature", props, required_hazard)

    big_event = first_feature(big_events_path)
    if not isinstance(big_event.get("properties"), dict):
        raise PipelineError("big_events feature missing properties")

    etas = load_json(etas_path)
    require_keys("etas_params", etas, {"canonical_etas", "rj_aftershock", "disclaimer"})
    canonical = etas["canonical_etas"]
    if not isinstance(canonical, dict):
        raise PipelineError("canonical_etas must be an object")
    require_keys("canonical_etas", canonical, {"source", "warning"})
    if "AFAD" not in str(canonical["source"]):
        raise PipelineError("canonical_etas source must reference AFAD")
    if "tahmini" not in str(etas["disclaimer"]).lower():
        raise PipelineError("etas_params disclaimer must state prediction limits")

    return {
        "source": summary["source"],
        "catalog_events": catalog_events,
        "hazard_cells": len(load_json(hazard_path)["features"]),
        "big_events": len(load_json(big_events_path)["features"]),
        "date_range": f"{summary['date_min']} → {summary['date_max']}",
    }


def read_catalog_magnitudes(path: Path) -> np.ndarray:
    """Read finite magnitudes from the processed catalog for completeness checks."""
    require_file(path)
    mags: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mag" not in reader.fieldnames:
            raise PipelineError(f"Catalog missing 'mag' column: {path}")
        for row in reader:
            raw = row.get("mag")
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if value == value:  # drop NaN
                mags.append(value)
    if not mags:
        raise PipelineError(f"Catalog has no usable magnitudes: {path}")
    return np.asarray(mags, dtype=float)


def assert_min_mag_complete(min_mag: float, mc: float) -> None:
    """Surface threshold must sit at or above completeness; else bias, not seismicity."""
    if min_mag < mc:
        raise PipelineError(
            f"min_mag ({min_mag}) Mc altında ({mc}); bu eşikte katalog tamamlanmamış, "
            "tehlike yüzeyi sismisiteyi değil raporlama yanlılığını modeller."
        )


def assert_gain_top25(gain: float | None) -> None:
    """Top-25% concentration gain must exceed 1.0 (area-uniform baseline).

    `gain_top25 = 1.0` means the highest-risk 25% of cells captured exactly their
    proportional share of future events — no better than area-uniform. `> 1.0` means
    those cells captured disproportionately more. `<= 1.0` is a complementary gate to
    `assert_beats_baseline`: even a positive overall ASS could coexist with a top-quartile
    that adds no concentration value. `None` means skill was unmeasurable (no test events).
    """
    if gain is None:
        raise PipelineError(
            "gain_top25 hesaplanamadı (test penceresinde olay yok); yüzey gain değerini gösteremiyor."
        )
    if float(gain) <= 1.0:
        raise PipelineError(
            f"gain_top25 ({gain}) <= 1.0: en riskli %25 alan gelecek olayları "
            "alan-uniform baseline'ı yenecek oranda toplamıyor; yüzey triage değeri yok."
        )


def assert_beats_baseline(area_skill_score: float | None) -> None:
    """The served surface must beat the area-uniform Poisson baseline.

    `area_skill_score > 0` means the highest-risk cells concentrate FUTURE
    (out-of-time, test-window) events better than a uniform-area baseline; `<= 0`
    means the surface adds no usable spatial information. `None` means skill was
    unmeasurable (no test events fell into trained cells). Either non-positive case
    must fail the gate: this is the machine-enforced form of the project's core
    "every model must beat a naive baseline" rule, applied to the served risk
    surface itself. The score is a DESCRIPTIVE skill measure — not a prediction.
    """
    if area_skill_score is None:
        raise PipelineError(
            "Uzaysal beceri ölçülemedi (test penceresinde eğitim hücrelerine düşen "
            "olay yok); yüzey baseline'ı yendiğini gösteremiyor."
        )
    if float(area_skill_score) <= 0.0:
        raise PipelineError(
            f"area_skill_score ({area_skill_score}) <= 0: yüzey alan-uniform Poisson "
            "baseline'ını YENMİYOR; bu yüzey ship edilemez (çekirdek baseline kuralı)."
        )


def assert_cell_deg_consistent(
    meta_cell_deg: float, polygon_edge_deg: float, tol: float = 1e-6
) -> None:
    """meta cell_deg and the exported polygon edge must describe the same grid."""
    if abs(float(meta_cell_deg) - float(polygon_edge_deg)) > tol:
        raise PipelineError(
            f"cell_deg tutarsız: meta {meta_cell_deg} vs poligon kenarı {polygon_edge_deg}"
        )


def polygon_edge_deg(feature: dict[str, Any]) -> float:
    """Longitude span (max(x) - min(x)) of the first polygon ring, rounded to 6."""
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise PipelineError("hazard feature missing geometry")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise PipelineError("hazard feature missing polygon coordinates")
    ring = coordinates[0]
    if not isinstance(ring, list) or len(ring) < 2:
        raise PipelineError("hazard feature has invalid polygon ring")
    xs = [float(point[0]) for point in ring]
    return round(max(xs) - min(xs), 6)


def validate_science(root: Path = ROOT) -> dict[str, Any]:
    """Numerical/scientific consistency gate over the exported hazard surface.

    Enforces what string/key checks cannot: (1) the surface threshold
    (meta.min_mag) is at or above magnitude-of-completeness Mc, (2) the meta grid
    size matches the exported polygon edge, (3) the catalog b-value is physically
    plausible (~1), (4) the displayed meta.mc matches the freshly-computed Mc
    (provenance drift guard), (5) the surface BEATS the area-uniform Poisson
    baseline — out-of-time area_skill_score recomputed from the real catalog at the
    SERVED params must be > 0, and the stamped meta.area_skill_score must match it
    (skill provenance drift guard). Any violation is a real scientific defect.

    Known limitation: min_mag is READ from meta and checked against Mc, but is
    NOT independently derived from the surface (unlike cell_deg, which is derived
    from the polygon edge). It is consistent today only because scripts/06 writes
    both meta and surface from the single MIN_MAG constant; a true surface-stamped
    min_mag is deferred to Faz-1 surface-versioning (end-to-end design doc,
    scoring-engine layer).
    """
    processed = root / PROCESSED_DIR
    web_data = root / WEB_DATA_DIR

    mags = read_catalog_magnitudes(processed / "turkey_catalog.csv")
    if mags.size < 2:
        raise PipelineError("Bilimsel kapı için yeterli büyüklük yok (n<2)")

    meta = load_json(web_data / "meta.json")
    require_keys(
        "meta",
        meta,
        {"min_mag", "cell_deg", "area_skill_score", "gain_top25", "n_test_events"},
    )
    min_mag = float(meta["min_mag"])
    cell_deg = float(meta["cell_deg"])

    mc = magnitude_of_completeness(mags)
    assert_min_mag_complete(min_mag, mc)

    feature = first_feature(web_data / "hazard_cells.geojson")
    assert_cell_deg_consistent(cell_deg, polygon_edge_deg(feature))

    bv = b_value_aki(mags, mc).b
    if not 0.6 <= bv <= 1.4:
        raise PipelineError(
            f"b-değeri ({bv}) makul aralık dışında (0.6–1.4); katalog ya da Mc şüpheli."
        )

    # meta.mc is a display/provenance value; catch silent drift from the real Mc.
    stamped_mc = meta.get("mc")
    if stamped_mc is not None and abs(float(stamped_mc) - mc) > 0.05:
        raise PipelineError(
            f"meta.mc ({stamped_mc}) katalogdan hesaplanan Mc ({mc}) ile uyuşmuyor "
            "(provenance drift)."
        )

    # The "beats baseline" gate: recompute the out-of-time spatial skill from the
    # real catalog at the SERVED params (meta.min_mag / meta.cell_deg) and enforce
    # area_skill_score > 0. Deterministic (no RNG). Then cross-check the stamped
    # meta value against the recompute (skill provenance drift), mirroring meta.mc.
    catalog_df = pd.read_csv(processed / "turkey_catalog.csv", parse_dates=["time"])
    skill = evaluate_surface_skill(catalog_df, TURKEY, min_mag=min_mag, cell_deg=cell_deg)
    area_skill_score = skill["area_skill_score"]
    assert_beats_baseline(area_skill_score)
    assert_gain_top25(skill["gain_top25"])

    stamped_ass = meta.get("area_skill_score")
    if stamped_ass is not None and abs(float(stamped_ass) - float(area_skill_score)) > 0.05:
        raise PipelineError(
            f"meta.area_skill_score ({stamped_ass}) katalogdan hesaplanan "
            f"({area_skill_score}) ile uyuşmuyor (skill provenance drift)."
        )

    return {
        "mc": mc,
        "b_value": bv,
        "min_mag": min_mag,
        "cell_deg": cell_deg,
        "area_skill_score": area_skill_score,
        "gain_top25": skill["gain_top25"],
        "n_test_events": skill["n_test_events"],
    }


def validate_served_surface(root: Path = ROOT, require_site_layer: bool = True) -> dict[str, Any]:
    """Catalog-INDEPENDENT contract gate over the committed web/data surface.

    The full `validate_outputs` + `validate_science` gates need the processed
    catalog and summary, which live under `data/processed/` and are GITIGNORED —
    so they are absent on a fresh CI checkout and cannot run on push. This gate
    validates only the COMMITTED web/data/* files, so it runs deterministically in
    CI with no network fetch and no catalog. It enforces, on every push, that the
    SERVED surface stays honest and on-contract:

      - source == AFAD, disclaimer states forecast limits ("tahmini"),
      - the NOT-FOR-PRICING E&O boundary (`assert_use_restriction`),
      - the stamped out-of-time `area_skill_score > 0` — i.e. the served surface
        CLAIMS to beat the area-uniform Poisson baseline (the full recompute of
        that claim from the real catalog is the local/release `--check` gate),
      - the hazard surface carries its full required property set,
      - etas_params honesty.

    Descriptive skill measure — not an earthquake prediction.
    """
    web_data = root / WEB_DATA_DIR

    meta = load_json(web_data / "meta.json")
    require_keys(
        "meta",
        meta,
        {
            "source",
            "disclaimer",
            "use_restriction",
            "mc",
            "area_skill_score",
            "gain_top25",
            "n_test_events",
            "min_mag",
            "cell_deg",
        },
    )
    if meta["source"] != "AFAD":
        raise PipelineError(f"meta source must be AFAD, got {meta['source']!r}")
    if "tahmini" not in str(meta["disclaimer"]).lower():
        raise PipelineError("meta disclaimer must state forecast/prediction limits")
    assert_use_restriction(meta)
    # The served surface's STAMPED "beats baseline" claim must be positive. The
    # recompute-from-catalog cross-check lives in validate_science (--check).
    assert_beats_baseline(meta["area_skill_score"])
    assert_gain_top25(meta["gain_top25"])

    feature = first_feature(web_data / "hazard_cells.geojson")
    props = feature.get("properties")
    if not isinstance(props, dict):
        raise PipelineError("hazard feature missing properties")
    required_hazard = {
        "risk_model",
        "risk_index",
        "annual_rate",
        "recent_rate",
        "recency_days",
        "energy_index",
        "shallow_fraction",
    }
    if require_site_layer:
        required_hazard |= {"site_risk", "site_class", "amp", "liq_class"}
    require_keys("hazard feature", props, required_hazard)

    etas = load_json(web_data / "etas_params.json")
    require_keys("etas_params", etas, {"canonical_etas", "rj_aftershock", "disclaimer"})
    canonical = etas["canonical_etas"]
    if not isinstance(canonical, dict):
        raise PipelineError("canonical_etas must be an object")
    if "AFAD" not in str(canonical.get("source", "")):
        raise PipelineError("canonical_etas source must reference AFAD")
    if "tahmini" not in str(etas["disclaimer"]).lower():
        raise PipelineError("etas_params disclaimer must state prediction limits")

    return {
        "source": meta["source"],
        "area_skill_score": meta["area_skill_score"],
        "gain_top25": meta["gain_top25"],
        "n_test_events": meta["n_test_events"],
        "hazard_cells": len(load_json(web_data / "hazard_cells.geojson")["features"]),
    }


def run_step(step: Step, root: Path = ROOT) -> None:
    print(f"\n==> {step.name}")
    print(" ".join(step.command))
    completed = subprocess.run(step.command, cwd=root, check=False)
    if completed.returncode != 0:
        raise PipelineError(f"Step failed: {step.name} (exit {completed.returncode})")


def print_summary(summary: dict[str, Any], options: PipelineOptions) -> None:
    print("\n=== Pipeline Summary ===")
    print(f"source: {summary['source']}")
    print(f"catalog_events: {summary['catalog_events']}")
    print(f"date_range: {summary['date_range']}")
    print(f"hazard_cells: {summary['hazard_cells']}")
    print(f"big_events: {summary['big_events']}")
    print(f"mc: {summary['mc']}")
    print(f"b_value: {summary['b_value']}")
    print(f"min_mag: {summary['min_mag']} (>= Mc)")
    print(f"cell_deg: {summary['cell_deg']}")
    print(
        f"area_skill_score: {summary['area_skill_score']} (> 0 -> baseline'i yener) | "
        f"gain_top25: {summary['gain_top25']} | n_test_events: {summary['n_test_events']}"
    )
    print(f"site_layer_required: {not options.skip_site_layer}")


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        steps = build_steps(options)
        if options.check_surface:
            print("Surface check mode: validating committed web/data contract (no catalog).")
        elif not steps:
            print("Check mode: validating existing outputs.")
        for step in steps:
            run_step(step)
        if options.check_surface:
            surface = validate_served_surface(
                ROOT, require_site_layer=not options.skip_site_layer
            )
            print("\n=== Served Surface Contract ===")
            print(f"source: {surface['source']}")
            print(f"hazard_cells: {surface['hazard_cells']}")
            print(
                f"area_skill_score: {surface['area_skill_score']} (> 0 -> baseline'i yener) | "
                f"gain_top25: {surface['gain_top25']} | n_test_events: {surface['n_test_events']}"
            )
        elif options.export_web or options.check:
            summary = validate_outputs(ROOT, require_site_layer=not options.skip_site_layer)
            summary.update(validate_science(ROOT))
            print_summary(summary, options)
        else:
            print("\nPipeline finished. Use --export-web to validate web outputs.")
    except PipelineError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if "site-layer" in str(exc) or "site_risk" in str(exc):
            print("Hint: rerun with --skip-site-layer if elevation data is unavailable.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
