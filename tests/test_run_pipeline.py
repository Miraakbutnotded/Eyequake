from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eyequake.config import TURKEY
from eyequake.forecast.spatial_skill import evaluate_surface_skill
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


def _hierarchical_catalog(seed: int = 0) -> pd.DataFrame:
    """Multi-cluster GR catalog with a clear spatial density gradient.

    Six hotspots of decreasing size (200…8) over distinct cells produce a surface
    whose high-risk cells genuinely concentrate future (test-window) events →
    out-of-time area_skill_score > 0 (the surface beats area-uniform Poisson).
    Magnitudes stay Gutenberg-Richter (b≈1, Mc≈2.35) so the Mc/b-value gates pass.
    """
    rng = np.random.default_rng(seed)
    centers = [(39.0, 28.0), (40.0, 30.0), (38.0, 38.0), (37.0, 36.0), (36.0, 30.0), (41.0, 40.0)]
    sizes = [200, 110, 60, 32, 16, 8]
    base = pd.Timestamp("2000-01-01", tz="UTC")
    span = (pd.Timestamp("2020-01-01", tz="UTC") - base).days
    rows = []
    for (clat, clon), sz in zip(centers, sizes):
        for _ in range(sz):
            off = rng.uniform(0.0, float(span))
            rows.append(
                (
                    base + pd.to_timedelta(off, unit="D"),
                    clat + rng.normal(0.0, 0.03),
                    clon + rng.normal(0.0, 0.03),
                )
            )
    rng.shuffle(rows)
    time, lat, lon = zip(*rows)
    return pd.DataFrame(
        {
            "time": list(time),
            "latitude": list(lat),
            "longitude": list(lon),
            "mag": _gr_catalog(len(rows), seed=seed + 5),
        }
    )


def _anti_correlated_catalog(seed: int = 0) -> pd.DataFrame:
    """GR catalog whose surface ANTI-correlates with the future → area_skill_score < 0.

    Training concentrates in hotspot A (plus a few seeds in B); the test window then
    fires in the low-risk hotspot B. The trained surface ranks A high, so future
    events land in low-ranked cells → negative skill. Mc/b-value stay valid so
    validate_science reaches the baseline gate (which must then REJECT this surface).
    """
    rng = np.random.default_rng(seed)
    a_lat, a_lon = 39.0, 28.0
    b_lat, b_lon = 37.0, 36.0
    rows = []
    for _ in range(120):  # train-heavy hotspot A (early)
        rows.append(
            (
                pd.Timestamp("2003-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 1000), unit="D"),
                a_lat + rng.normal(0, 0.05),
                a_lon + rng.normal(0, 0.05),
            )
        )
    for _ in range(8):  # a few early events in B so B is a trained (low-risk) cell
        rows.append(
            (
                pd.Timestamp("2004-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 500), unit="D"),
                b_lat + rng.normal(0, 0.05),
                b_lon + rng.normal(0, 0.05),
            )
        )
    for _ in range(60):  # test window fires in the low-risk hotspot B (late)
        rows.append(
            (
                pd.Timestamp("2018-01-01", tz="UTC") + pd.to_timedelta(rng.uniform(0, 600), unit="D"),
                b_lat + rng.normal(0, 0.05),
                b_lon + rng.normal(0, 0.05),
            )
        )
    time, lat, lon = zip(*rows)
    return pd.DataFrame(
        {
            "time": list(time),
            "latitude": list(lat),
            "longitude": list(lon),
            "mag": _gr_catalog(len(rows), seed=seed + 5),
        }
    )


def _write_science_root(
    tmp_path: Path, catalog: pd.DataFrame, *, min_mag: float = 2.5, cell_deg: float = 0.5
) -> Path:
    """Write a validate_science root: catalog + meta (skill stamped) + one hazard cell.

    The stamped area_skill_score/gain_top25/n_test_events are computed from the
    on-disk catalog exactly as validate_science recomputes them (same CSV round-trip),
    so the skill provenance cross-check matches by construction.
    """
    processed = tmp_path / "data" / "processed"
    web_data = tmp_path / "web" / "data"
    processed.mkdir(parents=True)
    web_data.mkdir(parents=True)

    catalog.to_csv(processed / "turkey_catalog.csv", index=False)
    on_disk = pd.read_csv(processed / "turkey_catalog.csv", parse_dates=["time"])
    skill = evaluate_surface_skill(on_disk, TURKEY, min_mag=min_mag, cell_deg=cell_deg)

    _write_json(
        web_data / "meta.json",
        {
            "source": "AFAD",
            "region": "turkey",
            "cell_deg": cell_deg,
            "min_mag": min_mag,
            "area_skill_score": skill["area_skill_score"],
            "gain_top25": skill["gain_top25"],
            "n_test_events": skill["n_test_events"],
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


def _science_root(tmp_path: Path, *, min_mag: float = 2.5, cell_deg: float = 0.5) -> Path:
    """Happy-path validate_science root: multi-cluster surface with positive skill."""
    return _write_science_root(
        tmp_path, _hierarchical_catalog(), min_mag=min_mag, cell_deg=cell_deg
    )


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

    assert set(result) == {
        "mc",
        "b_value",
        "min_mag",
        "cell_deg",
        "area_skill_score",
        "gain_top25",
        "n_test_events",
    }
    assert result["min_mag"] == 2.5
    assert result["cell_deg"] == 0.5
    assert result["min_mag"] >= result["mc"]
    assert 0.6 <= result["b_value"] <= 1.4
    # The surface must beat the area-uniform Poisson baseline (core project rule).
    assert result["area_skill_score"] > 0
    assert result["n_test_events"] > 0


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


# --- "Beats baseline" gate: out-of-time spatial skill must be positive ---


def test_assert_beats_baseline_passes_for_positive_skill():
    runner = _load_runner()
    runner.assert_beats_baseline(0.3)  # no raise: surface beats area-uniform Poisson


def test_assert_beats_baseline_raises_on_zero_negative_and_none():
    runner = _load_runner()
    for value in (0.0, -0.2, None):
        with pytest.raises(runner.PipelineError, match="baseline"):
            runner.assert_beats_baseline(value)


# --- gain_top25 > 1.0 gate ---


def test_assert_gain_top25_passes_above_one():
    runner = _load_runner()
    runner.assert_gain_top25(1.5)
    runner.assert_gain_top25(2.7)


def test_assert_gain_top25_raises_at_or_below_one():
    runner = _load_runner()
    for value in (1.0, 0.8, None):
        with pytest.raises(runner.PipelineError, match="gain_top25"):
            runner.assert_gain_top25(value)


def test_validate_served_surface_rejects_gain_at_or_below_one(tmp_path: Path):
    runner = _load_runner()
    root = _served_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["gain_top25"] = 0.95
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="gain_top25"):
        runner.validate_served_surface(root, require_site_layer=True)


def test_validate_science_returns_positive_skill(tmp_path: Path):
    runner = _load_runner()
    root = _science_root(tmp_path)

    result = runner.validate_science(root)

    # Recomputed from the catalog AND matches the stamped meta value (no drift).
    assert result["area_skill_score"] > 0
    meta = json.loads((root / "web" / "data" / "meta.json").read_text(encoding="utf-8"))
    assert result["area_skill_score"] == meta["area_skill_score"]


def test_validate_science_raises_when_surface_does_not_beat_baseline(tmp_path: Path):
    runner = _load_runner()
    # Anti-correlated surface: trained risk anti-correlates with the future →
    # area_skill_score < 0. Mc/b-value stay valid so the baseline gate is reached.
    root = _write_science_root(tmp_path, _anti_correlated_catalog())

    with pytest.raises(runner.PipelineError, match="baseline"):
        runner.validate_science(root)


def test_validate_science_requires_skill_keys_in_meta(tmp_path: Path):
    runner = _load_runner()
    root = _science_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["area_skill_score"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="area_skill_score"):
        runner.validate_science(root)


def test_validate_science_raises_when_stamped_skill_drifts(tmp_path: Path):
    runner = _load_runner()
    # Recomputed skill is ~0.37 and positive (gate passes); only the stamped
    # provenance value is corrupted far from it → skill provenance drift.
    root = _science_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["area_skill_score"] = 0.99
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="skill provenance"):
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


def test_assert_use_restriction_rejects_not_substring_hole_turkish():
    runner = _load_runner()
    # "fiyatlama notu" embeds "not" inside "notu" — the dropped English needle
    # let this affirmative misuse PASS. It must RAISE now.
    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.assert_use_restriction({"use_restriction": "fiyatlama notu: skor hazır"})


def test_assert_use_restriction_rejects_not_substring_hole_english():
    runner = _load_runner()
    # "rating note: ready for pricing" embeds "not" inside "note" — same hole,
    # and the text literally says ready-for-pricing. It must RAISE.
    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.assert_use_restriction(
            {"use_restriction": "rating note: ready for pricing"}
        )


def test_validate_outputs_rejects_missing_use_restriction(tmp_path: Path):
    runner = _load_runner()
    root = _minimal_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["use_restriction"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.validate_outputs(root, require_site_layer=True)


# --- Served-surface CI gate (catalog-independent committed web/data contract) ---


def _served_root(tmp_path: Path, *, area_skill_score: float = 0.4) -> Path:
    """Minimal COMMITTED web/data only (no catalog/summary) for validate_served_surface."""
    web_data = tmp_path / "web" / "data"
    web_data.mkdir(parents=True)

    _write_json(
        web_data / "meta.json",
        {
            "source": "AFAD",
            "region": "turkey",
            "cell_deg": 0.25,
            "min_mag": 3.0,
            "mc": 2.85,
            "area_skill_score": area_skill_score,
            "gain_top25": 2.7,
            "n_test_events": 15000,
            "n_cells": 1,
            "n_big_events": 0,
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
        web_data / "etas_params.json",
        {
            "canonical_etas": {"source": "AFAD kanonik katalog", "warning": "tanı göstergesi"},
            "rj_aftershock": {"model": "Reasenberg-Jones"},
            "disclaimer": "Kesin deprem tahmini DEĞİLDİR.",
        },
    )
    return tmp_path


def test_validate_served_surface_passes_on_committed_surface():
    """The REAL committed web/data must pass the served-surface contract (the CI gate)."""
    runner = _load_runner()
    repo_root = Path(__file__).resolve().parents[1]

    result = runner.validate_served_surface(repo_root, require_site_layer=True)

    assert result["source"] == "AFAD"
    # The served surface stamps a positive "beats baseline" claim.
    assert float(result["area_skill_score"]) > 0


def test_validate_served_surface_passes_on_minimal_root(tmp_path: Path):
    runner = _load_runner()
    root = _served_root(tmp_path)

    result = runner.validate_served_surface(root, require_site_layer=True)

    assert result["source"] == "AFAD"
    assert result["hazard_cells"] == 1


def test_validate_served_surface_rejects_non_positive_stamped_skill(tmp_path: Path):
    runner = _load_runner()
    # Stamped surface does NOT beat the area-uniform Poisson baseline → CI must fail.
    root = _served_root(tmp_path, area_skill_score=0.0)

    with pytest.raises(runner.PipelineError, match="baseline"):
        runner.validate_served_surface(root, require_site_layer=True)


def test_validate_served_surface_rejects_missing_use_restriction(tmp_path: Path):
    runner = _load_runner()
    root = _served_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["use_restriction"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="use_restriction"):
        runner.validate_served_surface(root, require_site_layer=True)


def test_validate_served_surface_requires_skill_keys(tmp_path: Path):
    runner = _load_runner()
    root = _served_root(tmp_path)
    meta_path = root / "web" / "data" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["gain_top25"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(runner.PipelineError, match="gain_top25"):
        runner.validate_served_surface(root, require_site_layer=True)


def test_build_steps_check_surface_returns_no_steps():
    runner = _load_runner()
    options = runner.PipelineOptions(source="afad", check_surface=True)

    assert runner.build_steps(options) == []
