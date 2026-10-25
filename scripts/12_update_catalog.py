"""AFAD olay arşivini artımlı günceller -> kanonik işlenmiş kataloğu yeniden kurar.

AFAD apiv2 yalnızca son ~11 günü servis ettiği için tarihsel katalog artık tek
seferde çekilemiyor (07_fetch_afad.py). Bu script düzenli koşar (GitHub Actions
cron, bkz. .github/workflows/data-refresh.yml): ufuk içindeki olayları çekip
kalıcı arşive ekler, kaçan aralıkları manifest'te `gaps` olarak kaydeder.

Servis edilen yüzey (06/10/11) ancak arşiv `--min-history-years` kadar geçmişi
kapsıyorsa yeniden üretilmeli; aksi halde birkaç günlük veriden kurulan bir harita
iyi yüzeyin üstüne yazılırdı. `--github-output` bu kararı `export_ready` olarak yazar.

Kullanım:
    ./.venv/bin/python scripts/12_update_catalog.py
    ./.venv/bin/python scripts/12_update_catalog.py --seed afad_export.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyequake.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_MIN_MAGNITUDE,
    PROCESSED_DIR,
    TURKEY,
)
from eyequake.data.archive import (  # noqa: E402
    apply_seed,
    history_years,
    load_archive,
    merge_events,
    plan_fetch,
    save_archive,
    update_manifest,
)
from eyequake.data.clean import clean_catalog, completeness_summary  # noqa: E402
from eyequake.data.fetch_afad import _normalize, fetch_window_afad  # noqa: E402

DEFAULT_ARCHIVE_DIR = DATA_DIR / "archive"


class UpdateError(RuntimeError):
    """Arşiv güvenle güncellenemedi (manifest'e dokunulmadı)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AFAD olay arşivini artımlı güncelle.")
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--archive-min-mag", type=float, default=0.0,
                   help="Arşive yazılacak en küçük büyüklük (hepsini sakla).")
    p.add_argument("--catalog-min-mag", type=float, default=DEFAULT_MIN_MAGNITUDE,
                   help="İşlenmiş kanonik katalog eşiği.")
    p.add_argument("--min-history-years", type=float, default=10.0,
                   help="Servis yüzeyini yeniden üretmek için gereken arşiv kapsamı.")
    p.add_argument("--seed", type=Path, default=None,
                   help="Tarihsel AFAD CSV (ortak şema ya da AFAD sütun adları).")
    p.add_argument("--skip-fetch", action="store_true", help="Yalnızca seed/yeniden kurulum.")
    p.add_argument("--github-output", action="store_true",
                   help="export_ready/history_years değerlerini $GITHUB_OUTPUT'a yaz.")
    return p.parse_args(argv)


def read_seed(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    if "eventID" in raw.columns:  # AFAD sütun adları → ortak şema
        return _normalize(raw.to_dict("records"))
    missing = {"id", "time", "latitude", "longitude", "mag"} - set(raw.columns)
    if missing:
        raise UpdateError(f"seed şeması tanınmadı, eksik sütunlar: {sorted(missing)}")
    return raw


def fetch_new(plan, min_mag: float) -> pd.DataFrame:
    new = fetch_window_afad(TURKEY, plan.start, plan.end, min_mag)
    span_days = (plan.end - plan.start).total_seconds() / 86_400.0
    # Türkiye'de M≥0'da günde onlarca olay olur; ≥1 günlük pencere boş geldiyse API
    # bozuk/değişmiş demektir — "kapsandı" diye işaretlemek sahte bir boşluk yaratırdı.
    if span_days >= 1.0 and new.empty:
        raise UpdateError(
            f"AFAD {span_days:.1f} günlük pencerede 0 olay döndü "
            f"({plan.start:%Y-%m-%d %H:%M} -> {plan.end:%Y-%m-%d %H:%M}); "
            "API değişmiş olabilir. Manifest güncellenmedi."
        )
    return new


def write_processed(events: pd.DataFrame, manifest: dict, processed_dir: Path,
                    min_mag: float) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    mags = pd.to_numeric(events["mag"], errors="coerce")
    clean = clean_catalog(events[mags >= min_mag])
    clean.to_csv(processed_dir / f"{TURKEY.name}_catalog.csv", index=False)
    summary = completeness_summary(clean)
    summary.update({
        "source": "AFAD",
        "min_mag": min_mag,
        "coverage_from": manifest.get("covered_from"),
        "coverage_until": manifest.get("covered_until"),
        "coverage_gaps": manifest.get("gaps", []),
        "history_years": round(history_years(manifest), 3),
    })
    (processed_dir / f"{TURKEY.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return summary


def run(args: argparse.Namespace, now: datetime | None = None, fetch=fetch_new) -> dict:
    now = now or datetime.now(timezone.utc)
    events, manifest = load_archive(args.archive_dir)
    n_before = len(events)

    if args.seed is not None:
        seed = read_seed(args.seed)
        times = pd.to_datetime(seed["time"], utc=True, format="ISO8601", errors="coerce")
        if times.isna().all():
            raise UpdateError("seed'de geçerli zaman yok")
        # Arşiv seed'den daha yeni revizyonlar taşır → arşiv sürümü kazanır.
        events = merge_events(seed, events)
        manifest = apply_seed(manifest, times.min().to_pydatetime(), times.max().to_pydatetime())

    n_new = 0
    if not args.skip_fetch:
        plan = plan_fetch(manifest, now)
        if plan.gap is not None:
            print(f"UYARI: {plan.gap[0]:%Y-%m-%d} -> {plan.gap[1]:%Y-%m-%d} aralığı API ufkunun "
                  "dışında kaldı; geri getirilemez, manifest'e boşluk olarak yazılıyor.")
        new = fetch(plan, args.archive_min_mag)
        merged = merge_events(events, new)
        n_new = len(merged) - len(events)
        events = merged
        manifest = update_manifest(manifest, plan, n_events=len(events), n_new=n_new, now=now)
    elif args.seed is None:
        raise UpdateError("--skip-fetch yalnızca --seed ile anlamlı")
    else:
        manifest["n_events"] = len(events)

    save_archive(args.archive_dir, events, manifest)
    summary = write_processed(events, manifest, args.processed_dir, args.catalog_min_mag)

    years = history_years(manifest)
    ready = years >= args.min_history_years
    print(f"Arşiv: {n_before} -> {len(events)} olay (+{n_new} yeni) | kapsam {years:.2f} yıl | "
          f"boşluk: {len(manifest.get('gaps', []))} | katalog M>={args.catalog_min_mag}: "
          f"{summary['n_events']} olay")
    print(f"export_ready={str(ready).lower()} (gerekli >= {args.min_history_years:g} yıl)")
    return {"export_ready": ready, "history_years": years, "n_new": n_new}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (UpdateError, RuntimeError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(f"export_ready={str(result['export_ready']).lower()}\n")
            fh.write(f"history_years={result['history_years']:.3f}\n")
            fh.write(f"n_new={result['n_new']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
