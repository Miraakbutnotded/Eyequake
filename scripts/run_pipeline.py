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
    args = parser.parse_args(argv)
    return PipelineOptions(
        source=args.source,
        skip_fetch=args.skip_fetch,
        skip_site_layer=args.skip_site_layer,
        export_web=args.export_web,
        check=args.check,
    )


def build_steps(options: PipelineOptions) -> list[Step]:
    if options.source != "afad":
        raise PipelineError(f"Unsupported source: {options.source}")
    if options.check:
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
    require_keys("meta", meta, {"source", "n_cells", "n_big_events", "disclaimer"})
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
    print(f"site_layer_required: {not options.skip_site_layer}")


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        steps = build_steps(options)
        if not steps:
            print("Check mode: validating existing outputs.")
        for step in steps:
            run_step(step)
        if options.export_web or options.check:
            summary = validate_outputs(ROOT, require_site_layer=not options.skip_site_layer)
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
