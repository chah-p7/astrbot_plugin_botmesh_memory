from __future__ import annotations

import unittest


class _Values:
    def __init__(self, **values: str):
        self.values = values

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)


class QueryCompatibilityTests(unittest.TestCase):
    def test_current_request_query_is_used(self) -> None:
        from astrbot_plugin_botmesh_memory.web_compat import query_value

        request = type("Request", (), {"query": _Values(scope_id="group-a")})()
        self.assertEqual(query_value(request, "scope_id"), "group-a")

    def test_legacy_request_args_is_used(self) -> None:
        from astrbot_plugin_botmesh_memory.web_compat import query_value

        request = type("Request", (), {"args": _Values(scope_id="group-b")})()
        self.assertEqual(query_value(request, "scope_id"), "group-b")

    def test_missing_query_returns_default(self) -> None:
        from astrbot_plugin_botmesh_memory.web_compat import query_value

        self.assertEqual(query_value(object(), "scope_id", "all"), "all")


if __name__ == "__main__":
    unittest.main()
