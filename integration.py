from __future__ import annotations

import inspect
import sys
from typing import Any


INTERFACE_NAME = "botmesh_memory"
BOTMESH_MEMORY_API_VERSION = 2
_provider: Any | None = None


def _matching_modules() -> list[Any]:
    current = sys.modules.get(__name__)
    return [
        module
        for name, module in list(sys.modules.items())
        if module is not None
        and (
            name == "astrbot_plugin_botmesh_memory.integration"
            or name.endswith(".astrbot_plugin_botmesh_memory.integration")
        )
        and module is not current
    ]


def _active_provider() -> Any | None:
    if _provider is not None:
        return _provider
    for module in _matching_modules():
        provider = getattr(module, "_provider", None)
        if provider is not None:
            return provider
    return None


def register_provider(provider: Any) -> None:
    global _provider
    _provider = provider
    for module in _matching_modules():
        setattr(module, "_provider", provider)


def unregister_provider(provider: Any) -> None:
    global _provider
    if _provider is provider:
        _provider = None
    for module in _matching_modules():
        if getattr(module, "_provider", None) is provider:
            setattr(module, "_provider", None)


async def record_exchange(
    *,
    umo: str,
    bot_id: str = "",
    logical_group_id: str = "",
    user_message: str = "",
    assistant_message: str,
    source_kind: str = "botmesh_direct",
    event: Any | None = None,
    extract: bool | None = None,
    summarize: bool | None = None,
) -> dict[str, Any]:
    provider = _active_provider()
    if provider is None:
        return {"success": False, "error": "provider_unavailable", "version": 2}
    method = getattr(provider, "record_external_exchange", None)
    if not callable(method):
        return {"success": False, "error": "api_unavailable", "version": 2}
    result = method(
        umo=umo,
        bot_id=bot_id,
        logical_group_id=logical_group_id,
        user_message=user_message,
        assistant_message=assistant_message,
        source_kind=source_kind,
        event=event,
        extract=extract,
        summarize=summarize,
    )
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("version", 2)
        return payload
    return {"success": bool(result), "version": 2}


async def get_context(
    *,
    umo: str,
    bot_id: str = "",
    logical_group_id: str = "",
    query: str = "",
    event: Any | None = None,
) -> dict[str, Any]:
    provider = _active_provider()
    if provider is None:
        return {}
    method = getattr(provider, "memory_context_payload", None)
    if not callable(method):
        return {}
    result = method(
        umo=umo,
        bot_id=bot_id,
        logical_group_id=logical_group_id,
        query=query,
        event=event,
    )
    if inspect.isawaitable(result):
        result = await result
    return dict(result) if isinstance(result, dict) else {}


def api_status() -> dict[str, Any]:
    provider = _active_provider()
    return {
        "name": INTERFACE_NAME,
        "version": BOTMESH_MEMORY_API_VERSION,
        "available": provider is not None,
        "capabilities": {
            "record_exchange": bool(
                provider is not None
                and callable(getattr(provider, "record_external_exchange", None))
            ),
            "get_context": bool(
                provider is not None
                and callable(getattr(provider, "memory_context_payload", None))
            ),
        },
    }


# AstrBot may load this file below a dynamic package namespace.  Register the
# canonical leaf and synchronize provider state across both module identities.
sys.modules.setdefault(
    "astrbot_plugin_botmesh_memory.integration", sys.modules[__name__]
)
