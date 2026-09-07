"""Lightweight opt-in debugging for relativistic analysis."""

from __future__ import annotations

import sys
from typing import Any


def debug_log(enabled: bool, component: str, message: str, **values: Any) -> None:
    """Emit a deterministic one-line diagnostic when debugging is enabled."""

    if not enabled:
        return
    suffix = ""
    if values:
        parts = []
        for key, value in values.items():
            parts.append(f"{key}={value}")
        suffix = " " + " ".join(parts)
    print(f"[tensor-toolkit:{component}] {message}{suffix}", file=sys.stderr)
