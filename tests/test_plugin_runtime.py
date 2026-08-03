from __future__ import annotations

import sys
import sqlite3
import tempfile
import time
import types
import unittest
from pathlib import Path


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _decorator(*_args, **_kwargs):
    return lambda function: function


def _command_group(_name):
    def decorate(function):
        function.command = lambda *_args, **_kwargs: _decorator()
        return function

    return decorate


class _Star:
    def __init__(self, context):
        self.context = context


class _StarTools:
    data_dir = Path(tempfile.gettempdir())

    @classmethod
    def get_data_dir(cls, _name):
        return cls.data_dir


class _Plain:
    def __init__(self, text=""):
        self.text = text


class _TextPart:
    def __init__(self, text=""):
        self.text = text

    def mark_as_temp(self):
        return self


class _Request:
    def __init__(self):
        self.payload = {}
        self.query = {}

    async def json(self, default=None):
        return self.payload if isinstance(self.payload, dict) else default


REQUEST = _Request()


class _ProviderRequest:
    def __init__(self, *, parts=None):
        self.system_prompt = "base"
        self.prompt = "hello"
        self.contexts = []
        self.extra_user_content_parts = list(parts or [])


class _Event:
    unified_msg_origin = "qq:GroupMessage:A_GROUP"

    def get_group_id(self):
        return "A_GROUP"

    def get_message_str(self):
        return "hello"

    def get_sender_id(self):
        return "user-1"

    def get_sender_name(self):
        return "小明"


def _install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []
    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.AstrBotConfig = dict
    api.logger = _Logger()

    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = types.SimpleNamespace(
        on_llm_request=_decorator,
        after_message_sent=_decorator,
        llm_tool=_decorator,
        permission_type=_decorator,
        command=_decorator,
        command_group=_command_group,
        PermissionType=types.SimpleNamespace(ADMIN="admin"),
    )

    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = _Plain
    provider = types.ModuleType("astrbot.api.provider")
    provider.ProviderRequest = object
    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = _Star
    star.StarTools = _StarTools
    web = types.ModuleType("astrbot.api.web")
    web.request = REQUEST
    web.json_response = lambda payload: payload
    web.error_response = lambda message, status_code=400: {
        "status": "error",
        "message": message,
        "status_code": status_code,
    }
    core = types.ModuleType("astrbot.core")
    core.__path__ = []
    agent = types.ModuleType("astrbot.core.agent")
    agent.__path__ = []
    message = types.ModuleType("astrbot.core.agent.message")
    message.TextPart = _TextPart

    astrbot.api = api
    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.message_components": components,
            "astrbot.api.provider": provider,
            "astrbot.api.star": star,
            "astrbot.api.web": web,
            "astrbot.core": core,
            "astrbot.core.agent": agent,
            "astrbot.core.agent.message": message,
        }
    )


_install_astrbot_stubs()

from astrbot_plugin_botmesh_memory import main as plugin_main


class _Config(dict):
    def save_config(self):
        return None


class _Context:
    def __init__(self):
        self.routes = []
        self.provider_manager = types.SimpleNamespace(
            providers_config=[
                {"id": "provider-a", "type": "openai_chat_completion", "model": "测试模型"}
            ]
        )
        self.last_llm_call = None

    def register_web_api(self, route, handler, methods, description):
        self.routes.append((route, handler, methods, description))

    async def get_current_chat_provider_id(self, _umo):
        return "provider-a"

    async def llm_generate(self, **kwargs):
        self.last_llm_call = kwargs
        return types.SimpleNamespace(
            completion_text=(
                '{"facts":[{"key":"plan","text":"周六一起吃饭",'
                '"basis":"user_explicit","confidence":0.95}],'
                '"corrections":[],"private_memories":[{"text":"不应保存"}],'
                '"episode":{"topic_key":"weekend-plan","title":"周末安排",'
                '"summary":"大家确认周六一起吃饭。","participants":["小明","小A"],'
                '"confirmed":["周六一起吃饭"],"unresolved":[]}}'
            )
        )


class _BotMeshLabels:
    def management_labels(self):
        return {
            "bots": {"bot_a": "小A"},
            "groups": {"main_group": "主群"},
            "scopes": {
                "botmesh:main_group": "主群",
                "qq:GroupMessage:A_GROUP": "主群",
            },
            "scope_groups": {
                "botmesh:main_group": "main_group",
                "qq:GroupMessage:A_GROUP": "main_group",
            },
            "bot_ids": {"bot_a": "bot_a", "a-account": "bot_a"},
            "memory_keys": {"main_group|bot_a": "小A角色"},
        }


class PluginPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        _StarTools.data_dir = Path(self.temporary.name)
        self.context = _Context()
        self.plugin = plugin_main.BotMeshMemoryPlugin(
            self.context,
            _Config(enabled=True, extraction_timeout_seconds=30),
        )
        from astrbot_plugin_botmesh import integration as botmesh_integration

        botmesh_integration.register_provider(_BotMeshLabels())

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temporary.cleanup()

    async def test_workspace_uses_human_labels_and_lists_empty_logical_groups(self):
        payload = self.plugin._workspace_payload()

        self.assertEqual(payload["labels"]["bots"]["bot_a"], "小A")
        self.assertEqual(payload["labels"]["scopes"]["botmesh:main_group"], "主群")
        self.assertEqual(
            payload["labels"]["scope_groups"]["qq:GroupMessage:A_GROUP"],
            "main_group",
        )
        self.assertEqual(
            payload["logical_groups"],
            [
                {
                    "logical_group_id": "main_group",
                    "scope_id": "botmesh:main_group",
                    "display_name": "主群",
                }
            ],
        )
        self.assertEqual(
            payload["logical_bots"],
            [{"bot_id": "bot_a", "display_name": "小A"}],
        )
        self.assertIn(
            "botmesh:main_group",
            {item["scope_id"] for item in payload["scopes"]},
        )
        self.assertTrue(
            any(route.endswith("/workspace/summarize") for route, *_ in self.context.routes)
        )

    async def test_workspace_enriches_dynamic_life_schedule_for_page(self):
        schedule_id = self.plugin.store.record_exchange(
            scope_id="botmesh:main_group",
            bot_id="bot_a",
            memory_key="小A角色",
            umo="",
            user_message="",
            assistant_message="📅 今日日程（2026-08-03）\n上午：整理书架",
            source_kind="dynamic_life_state",
            created_at=10,
        )

        payload = self.plugin._workspace_payload("botmesh:main_group")

        self.assertEqual(len(payload["schedules"]), 1)
        schedule = payload["schedules"][0]
        self.assertEqual(schedule["id"], schedule_id)
        self.assertEqual(schedule["business_date"], "2026-08-03")
        self.assertEqual(schedule["schedule_text"], schedule["assistant_message"])
        self.assertEqual(schedule["memory_key"], "小A角色")

    async def test_memory_injection_is_idempotent_and_skips_chathistory_overlap(self):
        async def payload(**_kwargs):
            return {
                "identity_block": "<botmesh_memory_identity>identity</botmesh_memory_identity>",
                "context_text": "<botmesh_memory_context>memory</botmesh_memory_context>",
                "raw_history_text": "<botmesh_recent_chat>raw</botmesh_recent_chat>",
                "scene_text": "<botmesh_scene>scene</botmesh_scene>",
                "recent_history": [{"content": "hello"}],
                "facts": [{}],
                "private_memories": [{}],
                "episodes": [{}],
                "scene": None,
            }

        self.plugin.memory_context_payload = payload
        self.plugin._resolve_scope = lambda **_kwargs: {
            "scope_id": "botmesh:main_group",
            "logical_group_id": "main_group",
            "bot_id": "bot_a",
            "memory_key": "小A角色",
        }
        request = _ProviderRequest(
            parts=[_TextPart("<group_chat_history>history</group_chat_history>")]
        )

        await self.plugin.inject_memory(_Event(), request)
        await self.plugin.inject_memory(_Event(), request)

        combined = self.plugin._request_context_text(request)
        self.assertEqual(combined.count("<botmesh_memory_identity>"), 1)
        self.assertEqual(combined.count("<current_sender "), 1)
        self.assertEqual(combined.count("<botmesh_memory_context>"), 1)
        self.assertEqual(combined.count("<botmesh_scene>"), 1)
        self.assertEqual(combined.count("<botmesh_recent_chat>"), 0)
        self.assertEqual(combined.count("<group_chat_history>"), 1)

    async def test_memory_raw_history_is_injected_once_without_chathistory(self):
        async def payload(**_kwargs):
            return {
                "identity_block": "",
                "context_text": "",
                "raw_history_text": "<botmesh_recent_chat>raw</botmesh_recent_chat>",
                "scene_text": "",
                "recent_history": [{"content": "hello"}],
                "facts": [],
                "private_memories": [],
                "episodes": [],
                "scene": None,
            }

        self.plugin.memory_context_payload = payload
        self.plugin._resolve_scope = lambda **_kwargs: {
            "scope_id": "botmesh:main_group",
            "logical_group_id": "main_group",
            "bot_id": "bot_a",
            "memory_key": "小A角色",
        }
        request = _ProviderRequest()

        await self.plugin.inject_memory(_Event(), request)
        await self.plugin.inject_memory(_Event(), request)

        combined = self.plugin._request_context_text(request)
        self.assertEqual(combined.count("<botmesh_recent_chat>"), 1)

    async def test_extraction_followup_is_appended_to_selected_memory(self):
        scope = {
            "scope_id": "botmesh:main_group",
            "logical_group_id": "main_group",
            "bot_id": "bot_a",
            "memory_key": "小A角色",
        }
        fact_id = self.plugin.store.upsert_fact(
            scope_id=scope["scope_id"],
            fact_key="weekend-plan",
            text="周六约饭",
            status="active",
            authority=80,
        )
        memory_id = self.plugin.store.add_private_memory(
            scope_id=scope["scope_id"],
            bot_id=scope["bot_id"],
            memory_key=scope["memory_key"],
            target_id="user-1",
            kind="commitment",
            topic_key="weekend-plan",
            text="答应会赴约",
            confidence=0.8,
            source_kind="test",
            source_id="seed",
        )

        self.plugin._apply_extraction(
            {
                "corrections": [],
                "facts": [
                    {
                        "fact_id": fact_id,
                        "operation": "append",
                        "key": "weekend-plan",
                        "text": "地点是车站旁的咖啡店",
                        "basis": "user_explicit",
                        "confidence": 0.9,
                    }
                ],
                "private_memories": [
                    {
                        "memory_id": memory_id,
                        "operation": "append",
                        "topic_key": "weekend-plan",
                        "target_id": "user-1",
                        "kind": "commitment",
                        "text": "还答应提前十分钟到",
                        "confidence": 0.9,
                    }
                ],
                "episode": {},
            },
            scope,
            "normal_reply",
            "exchange:2",
            "地点是车站旁的咖啡店",
            "好，我会提前到。",
        )

        workspace = self.plugin.store.workspace(scope["scope_id"])
        facts = {item["id"]: item for item in workspace["facts"]}
        private = {item["id"]: item for item in workspace["private_memories"]}
        self.assertEqual(
            facts[fact_id]["text"],
            "周六约饭；地点是车站旁的咖啡店",
        )
        self.assertEqual(
            private[memory_id]["text"],
            "答应会赴约；还答应提前十分钟到",
        )

    async def test_manual_summary_reads_history_and_never_writes_private_memory(self):
        history = types.ModuleType("astrbot_plugin_chat_history_context.integration")

        async def query_history(**_kwargs):
            now = time.time()
            return [
                {
                    "umo": "qq:GroupMessage:A_GROUP",
                    "ts": now - 30,
                    "sender_name": "小明",
                    "content": "周六一起吃饭吧。",
                },
                {
                    "umo": "qq:GroupMessage:B_GROUP",
                    "ts": now - 10,
                    "sender_name": "小A",
                    "content": "好，周六见。",
                },
            ]

        history.query_history = query_history
        sys.modules[history.__name__] = history
        REQUEST.payload = {
            "scope_id": "botmesh:main_group",
            "provider_id": "provider-a",
            "hours": 24,
            "limit": 160,
        }

        result = await self.plugin.page_summarize_workspace()

        self.assertTrue(result["summarized"])
        self.assertEqual(result["group_name"], "主群")
        self.assertEqual(result["source_message_count"], 2)
        self.assertIn("逻辑群名称：主群", self.context.last_llm_call["prompt"])
        workspace = self.plugin.store.workspace("botmesh:main_group")
        self.assertEqual(len(workspace["episodes"]), 1)
        self.assertEqual(len(workspace["facts"]), 1)
        self.assertEqual(workspace["private_memories"], [])

    async def test_page_imports_past_private_memory_with_role_key(self):
        REQUEST.payload = {
            "logical_group_id": "main_group",
            "import_kind": "private",
            "bot_id": "a-account",
            "text": "第一次见面时她递给我一杯热茶。\n我一直记得这件事。",
        }

        result = await self.plugin.page_import_workspace()

        self.assertEqual(result["imported"], 2)
        workspace = self.plugin.store.retrieve(
            scope_id="botmesh:main_group",
            bot_id="another-account",
            memory_key="小A角色",
            fact_limit=10,
            private_limit=10,
            episode_limit=10,
        )
        self.assertEqual(len(workspace["private_memories"]), 2)
        self.assertTrue(all(item["pinned"] for item in workspace["private_memories"]))

    async def test_config_fallback_maps_platform_account_and_inherits_global_persona(self):
        from astrbot_plugin_botmesh import integration as botmesh_integration

        botmesh_integration.unregister_provider(botmesh_integration._provider)
        config_path = Path(self.temporary.name) / "astrbot_plugin_botmesh_config.json"
        config_path.write_text(
            '{"bots":[{"bot_id":"bot_HASH","display_name":"QQ昵称",'
            '"account_id":"ACCOUNT","platform_id":"qq_main"}],'
            '"group_scopes":[{"group_id":"主群"}],'
            '"group_bindings":[{"group_id":"主群","bot_id":"bot_HASH",'
            '"platform_group_id":"RAW"}],'
            '"persona_profiles":[{"bot_id":"bot_HASH","group_id":"",'
            '"personality_prompt":"全局人格","memory_key":"角色名"},'
            '{"bot_id":"bot_HASH","group_id":"主群","worldview_prompt":"群世界观"}]}',
            encoding="utf-8",
        )

        labels = self.plugin._management_labels()
        scope = self.plugin._resolve_scope(
            umo="qq_main:GroupMessage:RAW",
            bot_id="ACCOUNT",
        )

        self.assertEqual(labels["bots"]["ACCOUNT"], "QQ昵称")
        self.assertEqual(labels["bots"]["HASH"], "QQ昵称")
        self.assertEqual(scope["logical_group_id"], "主群")
        self.assertEqual(scope["bot_id"], "bot_HASH")
        self.assertEqual(scope["memory_key"], "角色名")
        self.assertEqual(scope["identity_state"]["personality_prompt"], "全局人格")

    async def test_history_database_fallback_queries_logical_and_raw_group_rows(self):
        data_root = Path(self.temporary.name) / "data"
        self.plugin.data_dir = (
            data_root / "plugin_data" / "astrbot_plugin_botmesh_memory"
        )
        history_dir = data_root / "plugin_data" / "astrbot_plugin_chat_history_context"
        history_dir.mkdir(parents=True)
        config_dir = data_root / "config"
        config_dir.mkdir(parents=True)
        config_dir.joinpath("astrbot_plugin_botmesh_config.json").write_text(
            '{"bots":[{"bot_id":"bot_a","display_name":"小A",'
            '"platform_id":"qq_main"}],'
            '"group_scopes":[{"group_id":"主群"}],'
            '"group_bindings":[{"group_id":"主群","bot_id":"bot_a",'
            '"platform_group_id":"RAW"}]}',
            encoding="utf-8",
        )
        connection = sqlite3.connect(history_dir / "history.sqlite3")
        connection.executescript(
            """
            CREATE TABLE group_messages(
                id INTEGER PRIMARY KEY, umo TEXT, ts REAL, sender_id TEXT,
                sender_name TEXT, content TEXT, logical_group_id TEXT,
                logical_event_id TEXT, canonical_sender_id TEXT,
                source_bot_id TEXT
            );
            INSERT INTO group_messages VALUES(
                1, 'qq_main:GroupMessage:RAW', 10, 'u1', '甲', '旧作用域消息',
                '', 'event-1', 'user-1', '');
            INSERT INTO group_messages VALUES(
                2, 'another:GroupMessage:OTHER', 20, 'u2', '乙', '逻辑群消息',
                '主群', 'event-2', 'user-2', 'bot_a');
            INSERT INTO group_messages VALUES(
                3, 'qq_main:GroupMessage:RAW', 21, 'u2', '乙', '重复事件',
                '主群', 'event-2', 'user-2', 'bot_a');
            """
        )
        connection.commit()
        connection.close()

        rows = self.plugin._query_history_database(
            umo="",
            logical_group_id="主群",
            start_ts=0,
            end_ts=30,
            limit=20,
        )

        self.assertEqual([item["content"] for item in rows], ["旧作用域消息", "重复事件"])
        self.assertEqual(rows[0]["sender_name"], "甲")

    async def test_manual_summary_accepts_raw_scope_mapped_to_logical_group(self):
        history = types.ModuleType("astrbot_plugin_chat_history_context.integration")

        async def query_history(**kwargs):
            self.assertEqual(kwargs["logical_group_id"], "main_group")
            now = time.time()
            return [
                {
                    "umo": "qq:GroupMessage:A_GROUP",
                    "ts": now - 10,
                    "sender_name": "小明",
                    "content": "请记录这条群聊。",
                }
            ]

        history.query_history = query_history
        sys.modules[history.__name__] = history
        REQUEST.payload = {
            "scope_id": "qq:GroupMessage:A_GROUP",
            "provider_id": "provider-a",
        }

        result = await self.plugin.page_summarize_workspace()

        self.assertTrue(result["summarized"])
        self.assertEqual(result["group_name"], "主群")
        workspace = self.plugin.store.workspace("botmesh:main_group")
        self.assertEqual(len(workspace["episodes"]), 1)


if __name__ == "__main__":
    unittest.main()
