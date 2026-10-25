"""Bilimsel-dürüstlük değişmezleri (enforced) — tek doğruluk kaynağı.

docs/POSITIONING.md'deki yasaklı-iddia kuralını ve Track C reddini koda taşır.
"""

from __future__ import annotations

from eyequake.integrity.claims import (
    BANNED_CLAIMS,
    TRACK_C_STATUS,
    ClaimHit,
    scan_text,
)

__all__ = ["BANNED_CLAIMS", "TRACK_C_STATUS", "ClaimHit", "scan_text"]
