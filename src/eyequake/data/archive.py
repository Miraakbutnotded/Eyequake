"""Artımlı (incremental) AFAD olay arşivi — kaydırmalı API ufkuna karşı birikim.

AFAD apiv2 (2026-10 itibarıyla) yalnızca son ~11 günü servis ediyor; tarihsel
katalog tek seferde çekilemiyor. Çözüm: düzenli koşan bir güncelleyici her seferinde
ufuk içindeki olayları çekip kalıcı bir arşive ekler. Arşiv bu yüzden ufuktan daha
sık güncellendiği sürece boşluksuz büyür.

Dürüstlük kuralları:
  - Güncelleyici ufuktan uzun süre koşmadıysa kaçan aralık GERİ GETİRİLEMEZ; bu
    aralık manifest'te `gaps` olarak açıkça kaydedilir, sessizce yutulmaz.
  - Aynı olay AFAD'da sonradan revize edilebilir (büyüklük/konum); son çekilen
    sürüm kazanır.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ARCHIVE_FILE = "afad_events.csv.gz"
MANIFEST_FILE = "manifest.json"

# API ufku ~11 gün; 10 gün güvenli tarafta kalır.
DEFAULT_HORIZON_DAYS = 10.0
# Geç raporlanan / revize edilen olayları yakalamak için örtüşme.
DEFAULT_OVERLAP_DAYS = 2.0

_SCHEMA = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place"]


@dataclass(frozen=True)
class FetchPlan:
    start: datetime
    end: datetime
    gap: tuple[datetime, datetime] | None  # geri getirilemeyen kaçan aralık


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def plan_fetch(
    manifest: dict,
    now: datetime,
    *,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    overlap_days: float = DEFAULT_OVERLAP_DAYS,
) -> FetchPlan:
    """Bu koşuda çekilecek [start, end] aralığı ve varsa kaçan aralık.

    İlk koşuda (manifest boş) ufkun tamamı çekilir. Sonrakilerde `covered_until -
    overlap`'tan başlanır; bu nokta ufkun dışına düştüyse ufuk sınırına kırpılır ve
    arada kalan aralık `gap` olarak döner.
    """
    horizon_start = now - timedelta(days=horizon_days)
    covered_until = manifest.get("covered_until")
    if not covered_until:
        return FetchPlan(start=horizon_start, end=now, gap=None)
    until = _parse(covered_until)
    if until > now:
        raise ValueError(f"manifest covered_until ({covered_until}) gelecekte")
    start = until - timedelta(days=overlap_days)
    if start >= horizon_start:
        return FetchPlan(start=start, end=now, gap=None)
    gap = (until, horizon_start) if until < horizon_start else None
    return FetchPlan(start=horizon_start, end=now, gap=gap)


def merge_events(archive: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Arşive yeni olayları ekler; aynı id'de SON çekilen sürüm kazanır (revizyon)."""
    frames = [f.reindex(columns=_SCHEMA) for f in (archive, new) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=_SCHEMA)
    merged = pd.concat(frames, ignore_index=True)
    merged["id"] = merged["id"].astype(str)
    for col in ("latitude", "longitude", "depth", "mag"):  # CSV/seed metin gelebilir
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.drop_duplicates(subset="id", keep="last")
    order = pd.to_datetime(merged["time"], utc=True, format="ISO8601", errors="coerce")
    return merged.assign(_t=order).sort_values("_t").drop(columns="_t").reset_index(drop=True)


def update_manifest(
    manifest: dict, plan: FetchPlan, *, n_events: int, n_new: int, now: datetime
) -> dict:
    out = dict(manifest)
    out.setdefault("source", "AFAD")
    out.setdefault("covered_from", _iso(plan.start))
    out["covered_until"] = _iso(plan.end)
    gaps = list(out.get("gaps", []))
    if plan.gap is not None:
        gaps.append({"from": _iso(plan.gap[0]), "to": _iso(plan.gap[1])})
    out["gaps"] = gaps
    out["n_events"] = int(n_events)
    out["last_run"] = {"at": _iso(now), "fetched_from": _iso(plan.start), "n_new": int(n_new)}
    return out


def apply_seed(manifest: dict, seed_start: datetime, seed_end: datetime) -> dict:
    """Tarihsel bir seed kataloğunun kapsamını manifest'e işler.

    Seed, arşivin başından önce bitiyorsa aradaki aralık `gaps`'e eklenir (seed ile
    canlı arşiv arasında kimse veri toplamadı). `covered_from` seed başına çekilir.
    """
    out = dict(manifest)
    gaps = list(out.get("gaps", []))
    covered_from = out.get("covered_from")
    if covered_from and seed_end < _parse(covered_from):
        gaps.append({"from": _iso(seed_end), "to": covered_from})
    if not covered_from or seed_start < _parse(covered_from):
        out["covered_from"] = _iso(seed_start)
    if not out.get("covered_until"):
        out["covered_until"] = _iso(seed_end)
    out["gaps"] = sorted(gaps, key=lambda g: g["from"])
    out.setdefault("source", "AFAD")
    return out


def history_years(manifest: dict) -> float:
    """Arşivin kapsadığı süre (yıl), `covered_from`→`covered_until`."""
    if not manifest.get("covered_from") or not manifest.get("covered_until"):
        return 0.0
    span = _parse(manifest["covered_until"]) - _parse(manifest["covered_from"])
    return span.total_seconds() / (365.25 * 86_400.0)


def load_archive(archive_dir: Path) -> tuple[pd.DataFrame, dict]:
    archive_dir = Path(archive_dir)
    events_path = archive_dir / ARCHIVE_FILE
    manifest_path = archive_dir / MANIFEST_FILE
    events = (
        pd.read_csv(events_path, dtype={"id": str}) if events_path.exists()
        else pd.DataFrame(columns=_SCHEMA)
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    return events, manifest


def save_archive(archive_dir: Path, events: pd.DataFrame, manifest: dict) -> None:
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(archive_dir / ARCHIVE_FILE, index=False, compression="gzip")
    (archive_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
