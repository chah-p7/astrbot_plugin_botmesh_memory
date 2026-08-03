from __future__ import annotations

import json
import tempfile
import unittest
import sqlite3
from pathlib import Path

from astrbot_plugin_botmesh_memory.storage import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_workspace_exposes_only_dynamic_life_schedules_and_respects_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            schedule_id = store.record_exchange(
                scope_id="botmesh:group_a",
                bot_id="bot_a",
                memory_key="角色A",
                umo="",
                user_message="",
                assistant_message="📅 今日日程（2026-08-03）\n上午：看书",
                source_kind="dynamic_life_state",
                created_at=10,
            )
            store.record_exchange(
                scope_id="botmesh:group_a",
                bot_id="bot_a",
                memory_key="角色A",
                umo="umo",
                user_message="你好",
                assistant_message="你好。",
                source_kind="normal_reply",
                created_at=11,
            )
            store.record_exchange(
                scope_id="botmesh:group_b",
                bot_id="bot_b",
                memory_key="角色B",
                umo="",
                user_message="",
                assistant_message="📅 今日日程（2026-08-03）\n下午：散步",
                source_kind="dynamic_life_state",
                created_at=12,
            )

            all_rows = store.workspace()["schedules"]
            group_rows = store.workspace("botmesh:group_a")["schedules"]

            self.assertEqual(len(all_rows), 2)
            self.assertEqual([row["id"] for row in group_rows], [schedule_id])
            self.assertEqual(group_rows[0]["memory_key"], "角色A")
            self.assertEqual(group_rows[0]["source_kind"], "dynamic_life_state")
            self.assertNotIn("user_message", group_rows[0])

    def test_existing_database_adds_memory_key_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE private_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL, bot_id TEXT NOT NULL,
                    target_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL,
                    text TEXT NOT NULL, confidence REAL NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
                    source_kind TEXT NOT NULL DEFAULT '', source_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL, bot_id TEXT NOT NULL DEFAULT '',
                    umo TEXT NOT NULL DEFAULT '', user_message TEXT NOT NULL DEFAULT '',
                    assistant_message TEXT NOT NULL, source_kind TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                INSERT INTO private_memories(
                    scope_id, bot_id, target_id, kind, text, confidence,
                    pinned, status, source_kind, source_id, created_at, updated_at
                ) VALUES ('group', 'rev', 'owner', 'impression', '升级前记忆', 0.8,
                    0, 'active', 'test', '1', 1, 1);
                """
            )
            connection.commit()
            connection.close()

            store = MemoryStore(path)
            changed = store.adopt_legacy_memory_identity(
                scope_id="group", bot_id="rev", memory_key="蔚来"
            )
            payload = store.retrieve(
                scope_id="group",
                bot_id="other",
                memory_key="蔚来",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )

            self.assertEqual(changed, 1)
            self.assertEqual(payload["private_memories"][0]["text"], "升级前记忆")

    def test_higher_authority_fact_supersedes_and_lower_authority_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            low_id = store.upsert_fact(
                scope_id="group",
                fact_key="identity:rev",
                text="Rev 是莉芙",
                status="active",
                authority=20,
                now=1,
            )
            high_id = store.upsert_fact(
                scope_id="group",
                fact_key="identity:rev",
                text="Rev 当前由蔚来操控",
                status="active",
                authority=100,
                now=2,
            )
            conflict_id = store.upsert_fact(
                scope_id="group",
                fact_key="identity:rev",
                text="Rev 又变回莉芙",
                status="active",
                authority=20,
                now=3,
            )

            rows = {row["id"]: row for row in store.workspace("group")["facts"]}
            self.assertEqual(rows[low_id]["status"], "superseded")
            self.assertEqual(rows[high_id]["status"], "active")
            self.assertEqual(rows[conflict_id]["status"], "conflict")

    def test_private_memory_is_scoped_to_the_current_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            for bot_id, text in (("rev", "Rev 的印象"), ("tomo", "Tomo 的印象")):
                store.add_private_memory(
                    scope_id="group",
                    bot_id=bot_id,
                    target_id="user",
                    kind="impression",
                    text=text,
                    confidence=0.8,
                    source_kind="test",
                    source_id="1",
                )

            rev = store.retrieve(
                scope_id="group",
                bot_id="rev",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )
            self.assertEqual(
                [row["text"] for row in rev["private_memories"]],
                ["Rev 的印象"],
            )

    def test_memory_key_moves_subjective_memory_between_accounts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.add_private_memory(
                scope_id="group",
                bot_id="rev_account",
                memory_key="蔚来",
                target_id="owner",
                kind="impression",
                text="她答应过会等我",
                confidence=0.9,
                source_kind="test",
                source_id="1",
            )
            store.record_exchange(
                scope_id="group",
                bot_id="rev_account",
                memory_key="蔚来",
                umo="umo",
                user_message="在吗",
                assistant_message="我在。",
                source_kind="test",
            )

            moved = store.retrieve(
                scope_id="group",
                bot_id="other_account",
                memory_key="蔚来",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )
            other_role = store.retrieve(
                scope_id="group",
                bot_id="rev_account",
                memory_key="莉芙",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )
            self.assertEqual(
                [row["text"] for row in moved["private_memories"]],
                ["她答应过会等我"],
            )
            self.assertEqual(moved["self_exchanges"][0]["assistant_message"], "我在。")
            self.assertEqual(other_role["private_memories"], [])

    def test_legacy_bot_memory_is_adopted_by_first_explicit_role(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.add_private_memory(
                scope_id="group",
                bot_id="rev_account",
                target_id="owner",
                kind="impression",
                text="旧版账号记忆",
                confidence=0.8,
                source_kind="test",
                source_id="1",
            )
            changed = store.adopt_legacy_memory_identity(
                scope_id="group",
                bot_id="rev_account",
                memory_key="蔚来",
            )
            payload = store.retrieve(
                scope_id="group",
                bot_id="another_account",
                memory_key="蔚来",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )
            self.assertEqual(changed, 1)
            self.assertEqual(payload["private_memories"][0]["text"], "旧版账号记忆")

    def test_correction_supersedes_the_exact_old_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            old_id = store.upsert_fact(
                scope_id="group",
                fact_key="location",
                text="她们现在在教室",
                status="active",
                authority=80,
            )
            store.add_correction(
                scope_id="group",
                old_text="她们现在在教室",
                new_text="她们现在在天台",
                reason="用户纠正",
                source_kind="user_correction",
                source_id="2",
            )
            rows = {row["id"]: row for row in store.workspace("group")["facts"]}
            self.assertEqual(rows[old_id]["status"], "superseded")

    def test_episode_upsert_extends_time_range(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            first_id = store.upsert_episode(
                scope_id="group",
                episode_key="topic-1",
                title="话题",
                summary="第一阶段",
                participants=["A"],
                confirmed=["事实一"],
                unresolved=[],
                source_kind="test",
                source_id="1",
                started_at=10,
                ended_at=20,
                now=20,
            )
            second_id = store.upsert_episode(
                scope_id="group",
                episode_key="topic-1",
                title="话题",
                summary="完整阶段",
                participants=["A", "B"],
                confirmed=["事实一", "事实二"],
                unresolved=["待确认"],
                source_kind="test",
                source_id="2",
                started_at=5,
                ended_at=30,
                now=30,
            )
            episode = store.workspace("group")["episodes"][0]
            self.assertEqual(first_id, second_id)
            self.assertEqual(episode["started_at"], 5)
            self.assertEqual(episode["ended_at"], 30)
            self.assertEqual(episode["summary"], "第一阶段；完整阶段")
            self.assertEqual(json.loads(episode["participants_json"]), ["A", "B"])
            self.assertEqual(json.loads(episode["confirmed_json"]), ["事实一", "事实二"])

    def test_fact_followup_appends_to_existing_row(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            fact_id = store.upsert_fact(
                scope_id="group",
                fact_key="weekend-plan",
                text="周六约饭",
                status="active",
                authority=80,
                now=1,
            )
            merged_id = store.upsert_fact(
                scope_id="group",
                fact_key="weekend-plan",
                text="地点改在车站旁的咖啡店，时间是十二点",
                status="active",
                authority=80,
                merge_target_id=fact_id,
                merge_mode="append",
                now=2,
            )
            repeated_id = store.upsert_fact(
                scope_id="group",
                fact_key="weekend-plan",
                text="地点改在车站旁的咖啡店，时间是十二点",
                status="active",
                authority=80,
                merge_target_id=fact_id,
                merge_mode="append",
                now=3,
            )

            rows = store.workspace("group")["facts"]
            self.assertEqual(fact_id, merged_id)
            self.assertEqual(fact_id, repeated_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["text"],
                "周六约饭；地点改在车站旁的咖啡店；时间是十二点",
            )

    def test_private_followup_appends_but_current_emotion_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            memory_id = store.add_private_memory(
                scope_id="group",
                bot_id="bot",
                memory_key="role",
                target_id="owner",
                kind="commitment",
                topic_key="wait-promise",
                text="答应会等她",
                confidence=0.8,
                source_kind="test",
                source_id="1",
                now=1,
            )
            merged_id = store.add_private_memory(
                scope_id="group",
                bot_id="bot",
                memory_key="role",
                target_id="owner",
                kind="commitment",
                topic_key="wait-promise",
                text="还约定周六在车站见面",
                confidence=0.9,
                source_kind="test",
                source_id="2",
                now=2,
            )
            emotion_id = store.add_private_memory(
                scope_id="group",
                bot_id="bot",
                memory_key="role",
                target_id="owner",
                kind="emotion",
                topic_key="current-mood",
                text="现在有点担心",
                confidence=0.7,
                source_kind="test",
                source_id="3",
                now=3,
            )
            replaced_id = store.add_private_memory(
                scope_id="group",
                bot_id="bot",
                memory_key="role",
                target_id="owner",
                kind="emotion",
                topic_key="current-mood",
                text="现在已经安心了",
                confidence=0.9,
                source_kind="test",
                source_id="4",
                now=4,
            )

            rows = {row["id"]: row for row in store.workspace("group")["private_memories"]}
            self.assertEqual(memory_id, merged_id)
            self.assertEqual(
                rows[memory_id]["text"],
                "答应会等她；还约定周六在车站见面",
            )
            self.assertEqual(emotion_id, replaced_id)
            self.assertEqual(rows[emotion_id]["text"], "现在已经安心了")

    def test_migrate_legacy_records_maps_scope_platform_account_and_memory_key(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.add_private_memory(
                scope_id="qq_main:GroupMessage:RAW",
                bot_id="ACCOUNT",
                target_id="owner",
                kind="past",
                text="旧平台账号记忆",
                confidence=1.0,
                source_kind="test",
                source_id="1",
            )
            store.upsert_episode(
                scope_id="qq_main:GroupMessage:RAW",
                episode_key="old",
                title="过去",
                summary="旧群的情景",
                participants=[],
                confirmed=[],
                unresolved=[],
                source_kind="test",
                source_id="1",
                started_at=1,
                ended_at=2,
            )

            result = store.migrate_legacy_records(
                scope_groups={"qq_main:GroupMessage:RAW": "主群"},
                bot_ids={"ACCOUNT": "bot_a"},
                memory_keys={"主群|bot_a": "角色名"},
            )
            payload = store.retrieve(
                scope_id="botmesh:主群",
                bot_id="other",
                memory_key="角色名",
                fact_limit=10,
                private_limit=10,
                episode_limit=10,
            )

            self.assertEqual(result["moved_scope_rows"], 2)
            self.assertEqual(result["moved_identity_rows"], 1)
            self.assertTrue(Path(result["backup_path"]).is_file())
            self.assertEqual(payload["private_memories"][0]["bot_id"], "bot_a")
            self.assertEqual(payload["episodes"][0]["title"], "过去")


    def test_duplicate_compaction_never_crosses_logical_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.upsert_fact(
                scope_id="botmesh:one", fact_key="topic:a",
                text="大家约定周末一起看电影", status="inferred",
                authority=20, now=100,
            )
            store.upsert_fact(
                scope_id="botmesh:two", fact_key="topic:b",
                text="大家约定周末一起看电影", status="inferred",
                authority=20, now=100,
            )

            result = store.maintain(
                now=200,
                fact_ttl_days=0, superseded_fact_ttl_days=0,
                private_memory_ttl_days=0, episode_ttl_days=0,
                correction_ttl_days=0, exchange_ttl_days=0,
                scene_ttl_days=0, archive_ttl_days=0,
            )

            self.assertEqual(result["deduplicated"]["facts"], 0)
            self.assertEqual(len(store.workspace("botmesh:one")["facts"]), 1)
            self.assertEqual(len(store.workspace("botmesh:two")["facts"]), 1)

    def test_compaction_archives_duplicate_and_archive_can_be_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            first = store.upsert_fact(
                scope_id="botmesh:one", fact_key="plan:first",
                text="周六晚上一起看电影", status="active",
                authority=80, now=100,
            )
            second = store.upsert_fact(
                scope_id="botmesh:one", fact_key="plan:second",
                text="周六晚上一起看电影。", status="inferred",
                authority=20, now=90,
            )

            result = store.maintain(
                scope_id="botmesh:one", now=200,
                fact_ttl_days=0, superseded_fact_ttl_days=0,
                private_memory_ttl_days=0, episode_ttl_days=0,
                correction_ttl_days=0, exchange_ttl_days=0,
                scene_ttl_days=0, archive_ttl_days=0,
            )
            remaining = store.workspace("botmesh:one")["facts"]
            archived = store.list_archive(scope_id="botmesh:one")

            self.assertEqual(result["deduplicated"]["facts"], 1)
            self.assertEqual([row["id"] for row in remaining], [first])
            self.assertEqual(archived[0]["original_row_id"], second)
            restored = store.restore_archive(
                archived[0]["id"], scope_id="botmesh:one"
            )
            self.assertTrue(restored["success"])
            self.assertEqual(len(store.workspace("botmesh:one")["facts"]), 2)
            self.assertEqual(store.list_archive(scope_id="botmesh:one"), [])

    def test_ttl_preserves_pinned_and_high_authority_active_memories(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            old = 100.0
            now = old + 400 * 86400
            stale_fact = store.upsert_fact(
                scope_id="botmesh:one", fact_key="old:inferred",
                text="很久以前的低权威推测", status="inferred",
                authority=20, now=old,
            )
            protected_fact = store.upsert_fact(
                scope_id="botmesh:one", fact_key="old:explicit",
                text="用户明确说明的重要事实", status="active",
                authority=80, now=old,
            )
            pinned_fact = store.upsert_fact(
                scope_id="botmesh:one", fact_key="old:pinned",
                text="管理员固定的事实", status="inferred",
                authority=20, pinned=True, now=old,
            )
            stale_private = store.add_private_memory(
                scope_id="botmesh:one", bot_id="bot", memory_key="role",
                target_id="user", kind="impression", text="过期印象",
                confidence=0.5, source_kind="test", source_id="1", now=old,
            )
            pinned_private = store.add_private_memory(
                scope_id="botmesh:one", bot_id="bot", memory_key="role",
                target_id="user", kind="past", text="固定的过去记忆",
                confidence=1.0, source_kind="test", source_id="2",
                pinned=True, now=old,
            )
            store.record_exchange(
                scope_id="botmesh:one", bot_id="bot", memory_key="role",
                umo="umo", user_message="旧消息", assistant_message="旧回复",
                source_kind="test", created_at=old,
            )

            result = store.maintain(
                scope_id="botmesh:one", now=now, dedupe_enabled=False,
                fact_ttl_days=365, superseded_fact_ttl_days=90,
                protected_fact_authority=80, private_memory_ttl_days=365,
                episode_ttl_days=0, correction_ttl_days=0,
                exchange_ttl_days=30, scene_ttl_days=0, archive_ttl_days=0,
            )
            workspace = store.workspace("botmesh:one")
            fact_ids = {row["id"] for row in workspace["facts"]}
            private_ids = {row["id"] for row in workspace["private_memories"]}

            self.assertEqual(result["expired"]["facts"], 1)
            self.assertEqual(result["expired"]["private_memories"], 1)
            self.assertEqual(result["expired"]["exchanges"], 1)
            self.assertNotIn(stale_fact, fact_ids)
            self.assertIn(protected_fact, fact_ids)
            self.assertIn(pinned_fact, fact_ids)
            self.assertNotIn(stale_private, private_ids)
            self.assertIn(pinned_private, private_ids)
            self.assertEqual(len(store.list_archive(scope_id="botmesh:one")), 3)

    def test_maintenance_preview_does_not_modify_rows_or_status(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            store.record_exchange(
                scope_id="botmesh:one", bot_id="bot", umo="umo",
                user_message="old", assistant_message="old",
                source_kind="test", created_at=100,
            )
            preview = store.maintain(
                scope_id="botmesh:one", now=100 + 31 * 86400,
                dry_run=True, dedupe_enabled=False,
                fact_ttl_days=0, superseded_fact_ttl_days=0,
                private_memory_ttl_days=0, episode_ttl_days=0,
                correction_ttl_days=0, exchange_ttl_days=30,
                scene_ttl_days=0, archive_ttl_days=0,
            )

            self.assertEqual(preview["expired"]["exchanges"], 1)
            self.assertEqual(store.scope_counts(scope_id="botmesh:one")["exchanges"], 1)
            self.assertEqual(store.list_archive(scope_id="botmesh:one"), [])
            self.assertEqual(store.maintenance_status()["archive_count"], 0)


if __name__ == "__main__":
    unittest.main()
