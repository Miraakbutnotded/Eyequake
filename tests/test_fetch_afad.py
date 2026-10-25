"""fetch_afad.py: yeniden deneme + eksik-yıl fail-closed davranışı (ağsız, sahte oturum)."""

from __future__ import annotations

import json

import pytest
import requests

from eyequake.config import TURKEY
from eyequake.data import fetch_afad


class _Resp:
    def __init__(self, payload, status: int = 200):
        self.text = json.dumps(payload)
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _record(event_id: str, year: int) -> dict:
    return {
        "eventID": event_id, "date": f"{year}-06-01T00:00:00", "latitude": "39.0",
        "longitude": "35.0", "depth": "10", "magnitude": "4.1", "type": "ML",
        "location": "Test",
    }


class _Session:
    """Yıl → yanıt listesi; her çağrı sıradakini döner (istisna da olabilir)."""

    def __init__(self, script: dict[int, list]):
        self.script = script
        self.headers: dict = {}
        self.calls: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, timeout):
        year = int(url.split("start=")[1][:4])
        self.calls.append(year)
        nxt = self.script[year].pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def patch_session(monkeypatch):
    monkeypatch.setattr(fetch_afad.time, "sleep", lambda s: None)

    def install(script):
        session = _Session(script)
        monkeypatch.setattr(fetch_afad.requests, "Session", lambda: session)
        return session

    return install


def test_transient_failure_is_retried_not_dropped(patch_session):
    session = patch_session({
        2020: [requests.ConnectionError("reset"), _Resp([_record("a", 2020)])],
        2021: [_Resp([_record("b", 2021)])],
    })
    df = fetch_afad.fetch_catalog_afad(TURKEY, 2020, 2021, 2.5)
    assert sorted(df["id"]) == ["a", "b"]
    assert session.calls == [2020, 2020, 2021]


def test_persistently_missing_year_fails_closed(patch_session):
    """Bir yıl hiç gelmezse katalog SESSİZ boşlukla dönmemeli."""
    patch_session({
        2020: [_Resp([_record("a", 2020)])],
        2021: [_Resp([], status=503)] * 4,
    })
    with pytest.raises(RuntimeError, match="2021"):
        fetch_afad.fetch_catalog_afad(TURKEY, 2020, 2021, 2.5, max_tries=4)


def test_gaps_allowed_only_when_explicitly_requested(patch_session):
    patch_session({
        2020: [_Resp([_record("a", 2020)])],
        2021: [requests.Timeout("t")] * 2,
    })
    df = fetch_afad.fetch_catalog_afad(TURKEY, 2020, 2021, 2.5, max_tries=2, allow_gaps=True)
    assert df["id"].tolist() == ["a"]


def test_non_list_payload_is_treated_as_failure(patch_session):
    # AFAD hata gövdesini 200 + JSON nesne olarak dönebilir; olay listesi sanılmamalı.
    patch_session({2020: [_Resp({"message": "rate limited"})] * 2})
    with pytest.raises(RuntimeError, match="hiç veri"):
        fetch_afad.fetch_catalog_afad(TURKEY, 2020, 2020, 2.5, max_tries=2)


def test_empty_years_fail_closed_like_missing_years(patch_session):
    """HTTP 200 + [] başarı DEĞİL: geçmişi servis etmeyen API sahte katalog üretmemeli.

    Gerçek olay (2026-10): AFAD apiv2 yalnızca son ~11 günü döndürdü; 1990-2025 her
    yıl 0 olay geldi ve pipeline 45 olaylık bir 'kanonik katalog' yazdı.
    """
    patch_session({
        2024: [_Resp([])],
        2025: [_Resp([])],
        2026: [_Resp([_record("a", 2026)])],
    })
    with pytest.raises(RuntimeError, match="2024, 2025"):
        fetch_afad.fetch_catalog_afad(TURKEY, 2024, 2026, 2.5)


def test_empty_year_allowed_when_threshold_is_zero(patch_session):
    patch_session({2024: [_Resp([])], 2025: [_Resp([_record("a", 2025)])]})
    df = fetch_afad.fetch_catalog_afad(TURKEY, 2024, 2025, 6.5, min_events_per_year=0)
    assert df["id"].tolist() == ["a"]
