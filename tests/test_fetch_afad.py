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
