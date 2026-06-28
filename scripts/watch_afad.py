"""Poll AFAD for near-live events and write volatile web/data live files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.config import PROJECT_ROOT, TURKEY  # noqa: E402
from eyequake.data.fetch_afad import AFAD_URL  # noqa: E402

LIVE_EVENTS_PATH = Path("web/data/live_events.geojson")
LIVE_META_PATH = Path("web/data/live_meta.json")


class WatchError(RuntimeError):
    """Raised when the AFAD live watcher cannot fetch or write data."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch AFAD near-live earthquake events.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument("--min-mag", type=float, default=0.0, help="Minimum magnitude.")
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=120,
        help="AFAD time window to query on each poll.",
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=int,
        default=30,
        help="Mark feed stale when latest event is older than this.",
    )
    parser.add_argument("--once", action="store_true", help="Fetch once, write outputs, then exit.")
    return parser.parse_args(argv)


def build_afad_url(
    *,
    start: datetime,
    end: datetime,
    min_mag: float,
) -> str:
    params = {
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "minlat": TURKEY.min_lat,
        "maxlat": TURKEY.max_lat,
        "minlon": TURKEY.min_lon,
        "maxlon": TURKEY.max_lon,
        "minmag": min_mag,
    }
    query = "&".join(f"{key}={quote(str(value), safe=':-')}" for key, value in params.items())
    return f"{AFAD_URL}?{query}"


def fetch_recent_records(
    *,
    checked_at: datetime,
    lookback_minutes: int,
    min_mag: float,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    start = checked_at - timedelta(minutes=lookback_minutes)
    url = build_afad_url(start=start, end=checked_at, min_mag=min_mag)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "EyeQuake-live-watch/0.1"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.text.strip() else []
    except (requests.RequestException, ValueError) as exc:
        raise WatchError(f"AFAD live fetch failed: {exc}") from exc
    if not isinstance(payload, list):
        raise WatchError("AFAD live fetch returned non-list payload")
    return payload


def _event_time(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, format="mixed")


def _numeric(record: dict[str, Any], key: str) -> float:
    value = pd.to_numeric(record.get(key), errors="coerce")
    if pd.isna(value):
        raise WatchError(f"AFAD record has invalid {key}: {record.get(key)!r}")
    return float(value)


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        event_id = str(record.get("eventID") or f"missing-{index}")
        deduped[event_id] = record
    return list(deduped.values())


def build_live_payload(
    records: list[dict[str, Any]],
    *,
    checked_at: datetime,
    lookback_minutes: int,
    min_mag: float,
    stale_after_minutes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked_at = checked_at.astimezone(timezone.utc)
    rows = _dedupe_records(records)
    rows.sort(key=lambda record: _event_time(record["date"]), reverse=True)

    features: list[dict[str, Any]] = []
    latest_time: pd.Timestamp | None = None
    for record in rows:
        event_time = _event_time(record["date"])
        if latest_time is None or event_time > latest_time:
            latest_time = event_time
        lat = _numeric(record, "latitude")
        lon = _numeric(record, "longitude")
        mag = _numeric(record, "magnitude")
        depth = _numeric(record, "depth")
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
                "properties": {
                    "event_id": str(record.get("eventID", "")),
                    "time": event_time.isoformat(),
                    "mag": round(mag, 2),
                    "mag_type": record.get("type"),
                    "depth_km": round(depth, 2),
                    "place": str(record.get("location", "")),
                    "source": "AFAD",
                },
            }
        )

    lag_minutes: float | None = None
    latest_event_time: str | None = None
    status = "empty"
    if latest_time is not None:
        latest_dt = latest_time.to_pydatetime()
        lag_minutes = round((checked_at - latest_dt).total_seconds() / 60.0, 1)
        latest_event_time = latest_time.isoformat()
        status = "fresh" if lag_minutes <= stale_after_minutes else "stale"

    events = {"type": "FeatureCollection", "features": features}
    meta = {
        "source": "AFAD",
        "status": status,
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "lookback_minutes": lookback_minutes,
        "min_mag": min_mag,
        "stale_after_minutes": stale_after_minutes,
        "n_events": len(features),
        "latest_event_time": latest_event_time,
        "lag_minutes": lag_minutes,
        "disclaimer": (
            "AFAD yakın-canlı katalog verisidir; gecikme olabilir. "
            "Deprem tahmini değildir."
        ),
    }
    return events, meta


def write_live_outputs(root: Path, events: dict[str, Any], meta: dict[str, Any]) -> None:
    events_path = root / LIVE_EVENTS_PATH
    meta_path = root / LIVE_META_PATH
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc)
    records = fetch_recent_records(
        checked_at=checked_at,
        lookback_minutes=args.lookback_minutes,
        min_mag=args.min_mag,
    )
    events, meta = build_live_payload(
        records,
        checked_at=checked_at,
        lookback_minutes=args.lookback_minutes,
        min_mag=args.min_mag,
        stale_after_minutes=args.stale_after_minutes,
    )
    write_live_outputs(root, events, meta)
    print(
        f"AFAD live: {meta['status']} | events={meta['n_events']} | "
        f"latest={meta['latest_event_time']} | lag_min={meta['lag_minutes']}"
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.once:
            run_once(args)
            return 0
        while True:
            run_once(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped AFAD watcher.")
        return 0
    except WatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
