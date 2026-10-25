"""web/export.py model v2 tests: AFAD multi-signal hazard cell properties."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from eyequake.config import Region
from eyequake.web.export import (
    UNKNOWN_SOURCE,
    build_cell_index,
    cells_to_geojson,
    read_catalog_source,
)

REGION = Region("test", min_lat=38.0, max_lat=42.0, min_lon=30.0, max_lon=36.0)


def _multi_signal_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2010-01-01",
                    "2011-01-01",
                    "2012-01-01",
                    "2013-01-01",
                    "2024-01-01",
                    "2025-01-01",
                ],
                utc=True,
            ),
            "latitude": [39.1, 39.1, 39.1, 39.1, 40.6, 40.6],
            "longitude": [35.1, 35.1, 35.1, 35.1, 30.1, 30.1],
            "depth": [30.0, 25.0, 22.0, 20.0, 5.0, 7.0],
            "mag": [4.0, 4.1, 4.0, 4.2, 5.8, 5.9],
        }
    )


def test_build_cell_index_adds_afad_multi_signal_columns():
    cells = build_cell_index(_multi_signal_catalog(), REGION, cell_deg=0.5, min_mag=4.0)

    expected = {
        "risk_model",
        "risk_rate_component",
        "risk_magnitude_component",
        "risk_recent_component",
        "risk_energy_component",
        "risk_recency_component",
        "risk_shallow_component",
        "recent_rate",
        "recency_days",
        "energy_index",
        "shallow_fraction",
    }

    assert expected <= set(cells.columns)
    assert set(cells["risk_model"]) == {"v2_afad_multi_signal"}


def test_multi_signal_risk_can_prioritize_recent_large_shallow_cell():
    cells = build_cell_index(_multi_signal_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    top = cells.iloc[0]

    assert top["lat"] == 40.75
    assert top["lon"] == 30.25
    assert top["max_mag"] == 5.9
    assert top["shallow_fraction"] == 1.0


def test_cells_to_geojson_exports_model_v2_properties():
    cells = build_cell_index(_multi_signal_catalog(), REGION, cell_deg=0.5, min_mag=4.0)
    geojson = cells_to_geojson(cells, cell_deg=0.5)
    props = geojson["features"][0]["properties"]

    for key in [
        "risk_model",
        "recent_rate",
        "recency_days",
        "energy_index",
        "shallow_fraction",
    ]:
        assert key in props


def test_build_cell_index_handles_missing_depth_column():
    catalog = _multi_signal_catalog().drop(columns=["depth"])

    cells = build_cell_index(catalog, REGION, cell_deg=0.5, min_mag=4.0)

    assert "mean_depth" in cells.columns
    assert cells["shallow_fraction"].eq(0.0).all()


@pytest.mark.parametrize("source", ["AFAD", "USGS"])
def test_read_catalog_source_reports_the_recorded_source(tmp_path: Path, source):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"n_events": 3, "source": source}), encoding="utf-8")
    assert read_catalog_source(path) == source


@pytest.mark.parametrize(
    "content",
    [None, "{bozuk json", json.dumps({"n_events": 3}), json.dumps({"source": ""}), "[]"],
)
def test_read_catalog_source_never_defaults_to_afad(tmp_path: Path, content):
    """Kaynak bilinmiyorsa 'AFAD' DEĞİL bilinmeyen döner (provenance fail-closed).

    Eski davranış: özet `source` taşımıyorsa "AFAD" varsayılıyordu — USGS kataloğu
    (01_fetch_catalog.py `source` yazmıyordu) AFAD diye servis edilebiliyordu.
    """
    path = tmp_path / "summary.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert read_catalog_source(path) == UNKNOWN_SOURCE
    assert "AFAD" not in read_catalog_source(path)


def test_missing_depth_does_not_inflate_mean_depth():
    """Eksik derinlik 999 km sentinel'i olarak ortalamaya KARIŞMAMALI."""
    catalog = _multi_signal_catalog()
    catalog.loc[catalog.index[0], "depth"] = None
    cells = build_cell_index(catalog, REGION, cell_deg=1.0, min_mag=3.0)
    assert cells["mean_depth"].max() < 100.0


def test_cell_with_no_known_depth_serialises_as_null_not_nan():
    catalog = _multi_signal_catalog()
    catalog["depth"] = None
    cells = build_cell_index(catalog, REGION, cell_deg=1.0, min_mag=3.0)
    gj = cells_to_geojson(cells, cell_deg=1.0)
    text = json.dumps(gj, allow_nan=False)  # NaN olsaydı ValueError
    assert all(f["properties"]["mean_depth"] is None for f in json.loads(text)["features"])
    assert all(f["properties"]["shallow_fraction"] == 0.0 for f in gj["features"])


def test_zero_span_catalog_raises_instead_of_infinite_rate():
    catalog = _multi_signal_catalog().iloc[:1]
    with pytest.raises(ValueError, match="süresi"):
        build_cell_index(catalog, REGION, cell_deg=1.0, min_mag=3.0)
