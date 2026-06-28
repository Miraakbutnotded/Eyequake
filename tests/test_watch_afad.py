from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_watcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "watch_afad.py"
    spec = importlib.util.spec_from_file_location("watch_afad", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    event_id: str,
    date: str,
    magnitude: float,
    latitude: float = 39.0,
    longitude: float = 35.0,
) -> dict:
    return {
        "eventID": event_id,
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "depth": 7.3,
        "magnitude": magnitude,
        "type": "ML",
        "location": "Sındırgı (Balıkesir)",
    }


def test_build_live_payload_dedupes_sorts_and_marks_fresh():
    watcher = _load_watcher()
    checked_at = datetime(2026, 6, 27, 5, 1, tzinfo=timezone.utc)
    records = [
        _record("old", "2026-06-27 03:00:00", 1.2),
        _record("latest", "2026-06-27 04:56:00", 0.9),
        _record("latest", "2026-06-27 04:56:00", 0.9),
    ]

    events, meta = watcher.build_live_payload(
        records,
        checked_at=checked_at,
        lookback_minutes=120,
        min_mag=0.0,
        stale_after_minutes=30,
    )

    assert len(events["features"]) == 2
    assert events["features"][0]["properties"]["event_id"] == "latest"
    assert events["features"][0]["geometry"]["coordinates"] == [35.0, 39.0]
    assert events["features"][0]["properties"]["source"] == "AFAD"
    assert meta["status"] == "fresh"
    assert meta["n_events"] == 2
    assert meta["latest_event_time"] == "2026-06-27T04:56:00+00:00"
    assert meta["lag_minutes"] == 5.0


def test_build_live_payload_marks_stale_when_latest_event_old():
    watcher = _load_watcher()
    checked_at = datetime(2026, 6, 27, 5, 1, tzinfo=timezone.utc)
    records = [_record("old", "2026-06-27 03:53:55", 0.9)]

    _, meta = watcher.build_live_payload(
        records,
        checked_at=checked_at,
        lookback_minutes=120,
        min_mag=0.0,
        stale_after_minutes=30,
    )

    assert meta["status"] == "stale"
    assert meta["lag_minutes"] == 67.1


def test_build_live_payload_handles_empty_window():
    watcher = _load_watcher()
    checked_at = datetime(2026, 6, 27, 5, 1, tzinfo=timezone.utc)

    events, meta = watcher.build_live_payload(
        [],
        checked_at=checked_at,
        lookback_minutes=120,
        min_mag=0.0,
        stale_after_minutes=30,
    )

    assert events == {"type": "FeatureCollection", "features": []}
    assert meta["status"] == "empty"
    assert meta["latest_event_time"] is None
    assert meta["lag_minutes"] is None


def test_write_live_outputs_creates_web_data_files(tmp_path: Path):
    watcher = _load_watcher()
    events = {"type": "FeatureCollection", "features": []}
    meta = {"status": "empty"}

    watcher.write_live_outputs(tmp_path, events, meta)

    assert json.loads((tmp_path / "web/data/live_events.geojson").read_text()) == events
    assert json.loads((tmp_path / "web/data/live_meta.json").read_text()) == meta
