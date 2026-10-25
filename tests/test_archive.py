"""archive.py + 12_update_catalog.py: artımlı arşiv, boşluk kaydı, export kapısı (ağsız)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from eyequake.data.archive import (
    apply_seed,
    history_years,
    load_archive,
    merge_events,
    plan_fetch,
    save_archive,
    update_manifest,
)

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def _events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "time": [r[1] for r in rows],
            "latitude": 39.0,
            "longitude": 35.0,
            "depth": 10.0,
            "mag": [r[2] for r in rows],
            "magType": "ML",
            "place": "Test",
        }
    )


# --- plan_fetch ---


def test_first_run_fetches_the_whole_horizon():
    plan = plan_fetch({}, NOW, horizon_days=10)
    assert plan.start == NOW - timedelta(days=10)
    assert plan.end == NOW and plan.gap is None


def test_regular_run_overlaps_previous_coverage():
    manifest = {"covered_until": (NOW - timedelta(hours=6)).isoformat()}
    plan = plan_fetch(manifest, NOW, horizon_days=10, overlap_days=2)
    assert plan.start == NOW - timedelta(hours=6) - timedelta(days=2)
    assert plan.gap is None


def test_missed_runs_beyond_horizon_record_an_explicit_gap():
    """Ufuktan uzun süre koşulmadıysa kaçan aralık SESSİZCE yutulmaz."""
    until = NOW - timedelta(days=30)
    plan = plan_fetch({"covered_until": until.isoformat()}, NOW, horizon_days=10)
    assert plan.start == NOW - timedelta(days=10)
    assert plan.gap == (until, NOW - timedelta(days=10))


def test_coverage_in_the_future_is_rejected():
    with pytest.raises(ValueError, match="gelecekte"):
        plan_fetch({"covered_until": (NOW + timedelta(days=1)).isoformat()}, NOW)


# --- merge_events ---


def test_merge_dedupes_and_latest_revision_wins():
    old = _events([("a", "2026-10-20T00:00:00", 3.1), ("b", "2026-10-21T00:00:00", 2.0)])
    new = _events([("a", "2026-10-20T00:00:00", 3.4), ("c", "2026-10-19T00:00:00", 2.2)])
    out = merge_events(old, new)
    assert out["id"].tolist() == ["c", "a", "b"]  # zaman sıralı
    assert out.loc[out["id"] == "a", "mag"].item() == 3.4  # revizyon kazandı


def test_merge_of_empty_inputs_keeps_schema():
    out = merge_events(pd.DataFrame(), pd.DataFrame())
    assert out.empty and "id" in out.columns


# --- manifest ---


def test_manifest_accumulates_gaps_and_keeps_first_coverage_start():
    m1 = update_manifest({}, plan_fetch({}, NOW), n_events=5, n_new=5, now=NOW)
    later = NOW + timedelta(days=40)
    plan2 = plan_fetch(m1, later)
    m2 = update_manifest(m1, plan2, n_events=9, n_new=4, now=later)
    assert m2["covered_from"] == m1["covered_from"]
    assert len(m2["gaps"]) == 1
    assert m2["last_run"]["n_new"] == 4


def test_seed_extends_coverage_and_records_seed_to_archive_gap():
    manifest = {"covered_from": "2026-10-15T00:00:00+00:00",
                "covered_until": "2026-10-25T00:00:00+00:00", "gaps": []}
    out = apply_seed(
        manifest,
        datetime(1990, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert out["covered_from"].startswith("1990-01-01")
    assert out["gaps"] == [{"from": "2026-06-01T00:00:00+00:00",
                            "to": "2026-10-15T00:00:00+00:00"}]
    assert history_years(out) > 36


def test_archive_roundtrip(tmp_path: Path):
    events = _events([("a", "2026-10-20T00:00:00", 3.1)])
    save_archive(tmp_path, events, {"source": "AFAD", "gaps": []})
    loaded, manifest = load_archive(tmp_path)
    assert loaded["id"].tolist() == ["a"]
    assert manifest["source"] == "AFAD"


# --- 12_update_catalog.py run() ---


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "12_update_catalog.py"
    spec = importlib.util.spec_from_file_location("update_catalog", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **kw) -> argparse.Namespace:
    base = dict(
        archive_dir=tmp_path / "archive", processed_dir=tmp_path / "processed",
        archive_min_mag=0.0, catalog_min_mag=2.5, min_history_years=10.0,
        seed=None, skip_fetch=False, github_output=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_appends_new_events_and_is_idempotent(tmp_path: Path):
    mod = _load_script()
    batch = _events([("a", "2026-10-20T00:00:00", 3.1), ("b", "2026-10-21T00:00:00", 1.0)])
    r1 = mod.run(_args(tmp_path), now=NOW, fetch=lambda plan, mm: batch)
    r2 = mod.run(_args(tmp_path), now=NOW + timedelta(hours=6), fetch=lambda plan, mm: batch)
    assert r1["n_new"] == 2 and r2["n_new"] == 0
    summary = json.loads((tmp_path / "processed" / "turkey_summary.json").read_text("utf-8"))
    assert summary["source"] == "AFAD"
    assert summary["n_events"] == 1  # yalnızca M>=2.5 kanonik katalogda


def test_short_archive_is_not_export_ready(tmp_path: Path):
    """Birkaç günlük arşiv, iyi yüzeyin üstüne harita YAZDIRMAMALI."""
    mod = _load_script()
    batch = _events([("a", "2026-10-20T00:00:00", 3.1)])
    result = mod.run(_args(tmp_path), now=NOW, fetch=lambda plan, mm: batch)
    assert result["export_ready"] is False


def test_seeded_long_history_is_export_ready(tmp_path: Path):
    mod = _load_script()
    seed = tmp_path / "seed.csv"
    _events([("s1", "1995-01-01T00:00:00", 4.0), ("s2", "2026-10-10T00:00:00", 3.0)]).to_csv(
        seed, index=False
    )
    result = mod.run(_args(tmp_path, seed=seed), now=NOW,
                     fetch=lambda plan, mm: _events([("a", "2026-10-20T00:00:00", 3.1)]))
    assert result["export_ready"] is True
    _, manifest = load_archive(tmp_path / "archive")
    assert manifest["covered_from"].startswith("1995-01-01")


def test_empty_api_window_does_not_advance_coverage(tmp_path: Path, monkeypatch):
    """API 0 olay döndürürse manifest 'kapsandı' diye ilerlememeli (sahte boşluk)."""
    mod = _load_script()
    monkeypatch.setattr(mod, "fetch_window_afad", lambda *a, **k: _events([]))
    with pytest.raises(mod.UpdateError, match="0 olay"):
        mod.run(_args(tmp_path), now=NOW)
    assert not (tmp_path / "archive" / "manifest.json").exists()
