from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_root(tmp_path: Path) -> Path:
    processed = tmp_path / "data" / "processed"
    web_data = tmp_path / "web" / "data"
    processed.mkdir(parents=True)
    web_data.mkdir(parents=True)

    (processed / "turkey_catalog.csv").write_text(
        "time,latitude,longitude,mag\n"
        "2020-01-01T00:00:00+00:00,39.0,35.0,4.1\n",
        encoding="utf-8",
    )
    _write_json(
        processed / "turkey_summary.json",
        {
            "source": "AFAD",
            "n_events": 1,
            "date_min": "2020-01-01",
            "date_max": "2020-01-01",
        },
    )
    _write_json(
        web_data / "meta.json",
        {
            "source": "AFAD",
            "region": "turkey",
            "n_cells": 1,
            "n_big_events": 1,
            "disclaimer": "Göreli sismik tehlike indeksi; deprem tahmini DEĞİLDİR.",
        },
    )
    _write_json(
        web_data / "hazard_cells.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": []},
                    "properties": {
                        "risk_model": "v2_afad_multi_signal",
                        "risk_index": 42,
                        "annual_rate": 0.2,
                        "recent_rate": 0.1,
                        "recency_days": 12.0,
                        "energy_index": 7.2,
                        "shallow_fraction": 0.4,
                        "site_risk": 50,
                        "site_class": "C",
                        "amp": 1.2,
                        "liq_class": "orta",
                    },
                }
            ],
        },
    )
    _write_json(
        web_data / "big_events.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [35.0, 39.0]},
                    "properties": {"mag": 6.0},
                }
            ],
        },
    )
    _write_json(
        web_data / "etas_params.json",
        {
            "generated": "2026-06-26",
            "canonical_etas": {
                "source": "AFAD kanonik katalog",
                "branching_ratio": 0.8,
                "warning": "tanı göstergesi",
            },
            "rj_aftershock": {"model": "Reasenberg-Jones"},
            "disclaimer": "Kesin deprem tahmini DEĞİLDİR.",
        },
    )
    return tmp_path


def test_validate_outputs_returns_reproducibility_summary(tmp_path: Path):
    runner = _load_runner()
    root = _minimal_root(tmp_path)

    summary = runner.validate_outputs(root, require_site_layer=True)

    assert summary["source"] == "AFAD"
    assert summary["catalog_events"] == 1
    assert summary["hazard_cells"] == 1
    assert summary["big_events"] == 1
    assert summary["date_range"] == "2020-01-01 → 2020-01-01"


def test_validate_outputs_rejects_missing_site_layer(tmp_path: Path):
    runner = _load_runner()
    root = _minimal_root(tmp_path)
    hazard_path = root / "web" / "data" / "hazard_cells.geojson"
    hazard = json.loads(hazard_path.read_text(encoding="utf-8"))
    del hazard["features"][0]["properties"]["site_risk"]
    hazard_path.write_text(json.dumps(hazard), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="site_risk"):
        runner.validate_outputs(root, require_site_layer=True)


def test_validate_outputs_rejects_missing_v2_model_signal(tmp_path: Path):
    runner = _load_runner()
    root = _minimal_root(tmp_path)
    hazard_path = root / "web" / "data" / "hazard_cells.geojson"
    hazard = json.loads(hazard_path.read_text(encoding="utf-8"))
    del hazard["features"][0]["properties"]["energy_index"]
    hazard_path.write_text(json.dumps(hazard), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="energy_index"):
        runner.validate_outputs(root, require_site_layer=True)


def test_build_steps_respects_check_mode():
    runner = _load_runner()
    options = runner.PipelineOptions(source="afad", check=True, export_web=True)

    assert runner.build_steps(options) == []


def test_build_steps_includes_fetch_export_site_params_and_quality_gates():
    runner = _load_runner()
    options = runner.PipelineOptions(source="afad", export_web=True)

    names = [step.name for step in runner.build_steps(options)]

    assert names == [
        "fetch-afad",
        "export-web-data",
        "site-layer",
        "export-etas-params",
        "ruff",
        "pytest",
    ]


def test_build_steps_can_skip_fetch_and_site_layer():
    runner = _load_runner()
    options = runner.PipelineOptions(
        source="afad",
        skip_fetch=True,
        skip_site_layer=True,
        export_web=True,
    )

    names = [step.name for step in runner.build_steps(options)]

    assert "fetch-afad" not in names
    assert "site-layer" not in names
    assert "export-web-data" in names
