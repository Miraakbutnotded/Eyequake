from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyequake.web.export import build_cell_index


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
            "cell_deg": 0.25,
            "min_mag": 3.0,
            "mc": 2.5,
            "n_cells": 1,
            "n_big_events": 1,
            "disclaimer": "Göreli sismik tehlike indeksi; deprem tahmini DEĞİLDİR.",
            "use_restriction": (
                "Göreli triage / portföy sıralama. "
                "Premium rating için doğrudan girdi DEĞİLDİR."
            ),
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


# --- Scientific consistency gate (min_mag >= Mc, cell_deg, b-value) ---


def _gr_catalog(n: int = 4000, mc_true: float = 2.0, b: float = 1.0, seed: int = 0) -> np.ndarray:
    """Deterministic Gutenberg-Richter magnitudes (b≈1) for Mc/b-value tests."""
    rng = np.random.default_rng(seed)
    beta = b * np.log(10.0)
    mags = mc_true + rng.exponential(1.0 / beta, size=n)
    return np.round(mags, 1)


def _science_root(tmp_path: Path, *, min_mag: float = 2.5, cell_deg: float = 0.5) -> Path:
    """Root with a GR catalog + meta + one hazard cell whose polygon edge == cell_deg."""
    processed = tmp_path / "data" / "processed"
    web_data = tmp_path / "web" / "data"
    processed.mkdir(parents=True)
    web_data.mkdir(parents=True)

    mags = _gr_catalog()
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2000-01-01", periods=mags.size, freq="h", tz="UTC"),
            "latitude": 39.0,
            "longitude": 35.0,
            "mag": mags,
        }
    )
    frame.to_csv(processed / "turkey_catalog.csv", index=False)

    _write_json(
        web_data / "meta.json",
        {
            "source": "AFAD",
            "region": "turkey",
            "cell_deg": cell_deg,
            "min_mag": min_mag,
            "n_cells": 1,
            "n_big_events": 0,
            "disclaimer": "Göreli sismik tehlike indeksi; deprem tahmini DEĞİLDİR.",
            "use_restriction": (
                "Göreli triage / portföy sıralama. "
                "Premium rating için doğrudan girdi DEĞİLDİR."
            ),
        },
    )
    half = cell_deg / 2.0
    lon, lat = 35.0, 39.0
    ring = [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]
    _write_json(
        web_data / "hazard_cells.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"risk_index": 50},
                }
            ],
        },
    )
    return tmp_path


def test_build_cell_index_requires_explicit_min_mag():
    """min_mag must be keyword-only with NO default (silent 4.0 default was a latent bug)."""
    param = inspect.signature(build_cell_index).parameters["min_mag"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_assert_min_mag_complete_passes_when_at_or_above_mc():
    runner = _load_runner()
    # No raise when surface threshold is at/above completeness.
    runner.assert_min_mag_complete(3.0, 2.7)
    runner.assert_min_mag_complete(2.7, 2.7)


def test_assert_min_mag_complete_raises_when_below_mc():
    runner = _load_runner()
    with pytest.raises(runner.PipelineError, match="Mc"):
        runner.assert_min_mag_complete(2.0, 2.7)


def test_assert_cell_deg_consistent_passes_when_equal():
    runner = _load_runner()
    runner.assert_cell_deg_consistent(0.25, 0.25)


def test_assert_cell_deg_consistent_raises_when_different():
    runner = _load_runner()
    with pytest.raises(runner.PipelineError, match="cell_deg"):
        runner.assert_cell_deg_consistent(0.25, 0.50)


def test_polygon_edge_deg_returns_ring_width():
    runner = _load_runner()
    feature = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [35.0, 39.0],
                    [35.5, 39.0],
                    [35.5, 39.5],
                    [35.0, 39.5],
                    [35.0, 39.0],
                ]
            ],
        }
    }
    assert runner.polygon_edge_deg(feature) == 0.5


def test_validate_science_returns_mc_and_b_value(tmp_path: Path):
    runner = _load_runner()
    root = _science_root(tmp_path)

    result = runner.validate_science(root)

    assert set(result) == {"mc", "b_value", "min_mag", "cell_deg"}
    assert result["min_mag"] == 2.5
    assert result["cell_deg"] == 0.5
    assert result["min_mag"] >= result["mc"]
    assert 0.6 <= result["b_value"] <= 1.4


def test_validate_science_raises_when_min_mag_below_mc(tmp_path: Path):
    runner = _load_runner()
    root = _science_root(tmp_path, min_mag=1.0)

    with pytest.raises(runner.PipelineError, match="Mc"):
        runner.validate_science(root)


def test_validate_science_raises_when_cell_deg_mismatch(tmp_path: Path):
    runner = _load_runner()
    # meta says 0.5 but polygon is drawn at 0.5; corrupt meta cell_deg to 0.25.
    root = _science_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["cell_deg"] = 0.25
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="cell_deg"):
        runner.validate_science(root)


def test_validate_science_raises_when_stamped_mc_drifts(tmp_path: Path):
    runner = _load_runner()
    # min_mag stays valid; only the displayed meta.mc is corrupted to a wrong value.
    root = _science_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mc"] = 5.0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="provenance"):
        runner.validate_science(root)


# --- NOT-FOR-PRICING usage restriction gate (E&O / liability guard) ---


def test_assert_use_restriction_passes_for_not_for_rating_string():
    runner = _load_runner()
    # The real export string: relative triage, explicitly NOT a rating input.
    runner.assert_use_restriction(
        {
            "use_restriction": (
                "Göreli triage / portföy sıralama / accumulation control. "
                "Absolute loss'a kalibre DEĞİL — premium rating için doğrudan girdi DEĞİLDİR."
            )
        }
    )


def test_assert_use_restriction_raises_when_absent():
    runner = _load_runner()
    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.assert_use_restriction({})


def test_assert_use_restriction_raises_on_affirmative_misuse():
    runner = _load_runner()
    # Rating marker present but NO negation: implies the score IS ready for pricing.
    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.assert_use_restriction({"use_restriction": "Fiyatlama için hazır"})


def test_validate_outputs_rejects_missing_use_restriction(tmp_path: Path):
    runner = _load_runner()
    root = _minimal_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["use_restriction"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.validate_outputs(root, require_site_layer=True)
