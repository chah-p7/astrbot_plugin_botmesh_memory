from __future__ import annotations

from typing import Any


def query_value(request_obj: Any, key: str, default: str = "") -> Any:
    """Read a query value across AstrBot's legacy and current Web APIs."""
    for attribute in ("query", "args"):
        values = getattr(request_obj, attribute, None)
        getter = getattr(values, "get", None)
        if callable(getter):
            return getter(key, default)
    return default
