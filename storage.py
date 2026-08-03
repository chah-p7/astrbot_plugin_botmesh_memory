from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_STOPWORDS = frozenset(
    """
    的 了 吗 呢 吧 啊 呀 哦 嗯 哈 是 在 你 我 他 她 它 们 这 那 什么 怎么
    为什么 没有 有 不 就 都 也 还 很 和 与 或 一个 一下 现在 刚才 刚刚 最近
    大家 群里 群 请问 能 可以 知道 觉得 想 说 看 听 来 去 跟 给 把 被 让
    还是 是不是 有没有 谁 哪 哪些 时候 事情 东西 话题 消息 记录 今天 昨天
    """.split()
)


def _query_tokens(query: str) -> list[str]:
    """把查询文本拆成用于相关性的词元（拉丁词 + 中文二元组/短查询单字）。"""
    raw = str(query or "").strip().casefold()
    if not raw:
        return []
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9_]+", raw):
        if len(word) >= 2 and word not in _STOPWORDS:
            tokens.append(word)
    chars = [char for char in raw if "\u4e00" <= char <= "\u9fff"]
    if len(chars) <= 3:
        tokens.extend(char for char in chars if char not in _STOPWORDS)
    tokens.extend(
        bigram
        for bigram in (
            "".join(pair) for pair in zip(chars, chars[1:])
        )
        if bigram not in _STOPWORDS
        and not all(char in _STOPWORDS for char in bigram)
    )
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _text_relevance(text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    haystack = str(text or "").casefold()
    score = 0.0
    for token in tokens:
        count = haystack.count(token)
        if count:
            score += 1.0 + min(count, 3) * 0.5
    return score


def _text_bigrams(text: str) -> set[str]:
    cleaned = re.sub(
        r"[\s，。！？、；：“”\"'‘’（）()【】\-_…]",
        "",
        str(text or ""),
    ).casefold()
    if len(cleaned) < 4:
        return set()
    return {cleaned[index : index + 2] for index in range(len(cleaned) - 1)}


def _text_similarity(a: str, b: str) -> float:
    """基于二元组 Jaccard/包含度估算两条文本的相似度（0~1）。"""
    bigrams_a = _text_bigrams(a)
    bigrams_b = _text_bigrams(b)
    if not bigrams_a or not bigrams_b:
        return 0.0
    inter = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    if not union:
        return 0.0
    jaccard = inter / union
    containment = max(inter / len(bigrams_a), inter / len(bigrams_b))
    return max(jaccard, containment)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or "")).casefold()


def _merge_memory_text(existing: str, incoming: str, *, max_chars: int) -> tuple[str, bool]:
    """Append only novel clauses; return (merged_text, changed)."""
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not incoming:
        return existing[:max_chars], False
    if not existing:
        return incoming[:max_chars], True
    existing_normalized = _normalized_text(existing)
    incoming_normalized = _normalized_text(incoming)
    if (
        not incoming_normalized
        or incoming_normalized == existing_normalized
        or incoming_normalized in existing_normalized
        or _text_similarity(existing, incoming) >= 0.9
    ):
        return existing[:max_chars], False

    existing_clauses = [
        clause.strip()
        for clause in re.split(r"[\n，,。！？!?；;]+", existing)
        if clause.strip()
    ]
    novel: list[str] = []
    for clause in re.split(r"[\n，,。！？!?；;]+", incoming):
        clause = clause.strip()
        normalized = _normalized_text(clause)
        if not normalized or normalized in existing_normalized:
            continue
        if any(
            _text_similarity(clause, previous) >= 0.86
            for previous in (*existing_clauses, *novel)
        ):
            continue
        novel.append(clause)
    if not novel:
        return existing[:max_chars], False
    merged = existing.rstrip("\n，,。！？!?；; ") + "；" + "；".join(novel)
    return merged[:max_chars], merged[:max_chars] != existing[:max_chars]


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _merge_string_lists(*values: Any, limit: int = 30) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _json_string_list(value):
            normalized = _normalized_text(item)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(item)
            if len(result) >= max(1, int(limit)):
                return result
    return result


def _dedupe_similar(
    rows: list[sqlite3.Row],
    *,
    fields: tuple[str, ...],
    limit: int,
    threshold: float = 0.68,
) -> list[sqlite3.Row]:
    """按最佳排序保留每个相似簇的第一条，避免重复记忆挤占上下文名额。"""
    kept: list[sqlite3.Row] = []
    for row in rows:
        text = " ".join(str(row[field] or "") for field in fields)
        duplicate = any(
            _text_similarity(
                text,
                " ".join(str(previous[field] or "") for field in fields),
            )
            >= threshold
            for previous in kept
        )
        if not duplicate:
            kept.append(row)
        if len(kept) >= max(1, int(limit)):
            break
    return kept


def _retain_relevant(
    rows: list[sqlite3.Row],
    *,
    fields: tuple[str, ...],
    tokens: list[str],
    fallback: int,
) -> list[sqlite3.Row]:
    """Avoid filling a query-specific context with unrelated rows."""
    if not tokens:
        return rows
    relevant = [
        row
        for row in rows
        if (
            ("pinned" in row.keys() and int(row["pinned"] or 0) > 0)
            or _text_relevance(
                " ".join(str(row[field] or "") for field in fields),
                tokens,
            )
            > 0
        )
    ]
    if len(relevant) >= max(1, int(fallback)):
        return relevant
    selected_ids = {int(row["id"]) for row in relevant}
    result = list(relevant)
    for row in rows:
        if int(row["id"]) in selected_ids:
            continue
        result.append(row)
        if len(result) >= max(1, int(fallback)):
            break
    return result


def _limit_per_group(
    rows: list[sqlite3.Row],
    *,
    fields: tuple[str, ...],
    per_group: int,
) -> list[sqlite3.Row]:
    counts: dict[tuple[str, ...], int] = {}
    result: list[sqlite3.Row] = []
    for row in rows:
        key = tuple(str(row[field] or "").casefold() for field in fields)
        pinned = "pinned" in row.keys() and int(row["pinned"] or 0) > 0
        if not pinned and counts.get(key, 0) >= max(1, int(per_group)):
            continue
        counts[key] = counts.get(key, 0) + 1
        result.append(row)
    return result


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    predicate TEXT NOT NULL DEFAULT '',
                    object_text TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'inferred',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    authority INTEGER NOT NULL DEFAULT 20,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    source_kind TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    source_excerpt TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_facts_scope_status
                    ON facts(scope_id, status, pinned, authority, updated_at);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_key
                    ON facts(scope_id, fact_key, status);

                CREATE TABLE IF NOT EXISTS private_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    bot_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL DEFAULT '',
                    target_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL DEFAULT 'impression',
                    topic_key TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_kind TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_private_memory_scope_bot
                    ON private_memories(scope_id, bot_id, status, pinned, updated_at);

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    episode_key TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL,
                    participants_json TEXT NOT NULL DEFAULT '[]',
                    confirmed_json TEXT NOT NULL DEFAULT '[]',
                    unresolved_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    source_kind TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(scope_id, episode_key)
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_scope_time
                    ON episodes(scope_id, status, ended_at);

                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    old_text TEXT NOT NULL DEFAULT '',
                    new_text TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    source_kind TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_corrections_scope_time
                    ON corrections(scope_id, created_at);

                CREATE TABLE IF NOT EXISTS exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    bot_id TEXT NOT NULL DEFAULT '',
                    memory_key TEXT NOT NULL DEFAULT '',
                    umo TEXT NOT NULL DEFAULT '',
                    user_message TEXT NOT NULL DEFAULT '',
                    assistant_message TEXT NOT NULL,
                    source_kind TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_exchanges_scope_bot_time
                    ON exchanges(scope_id, bot_id, created_at);

                CREATE TABLE IF NOT EXISTS scene_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL DEFAULT '',
                    mood TEXT NOT NULL DEFAULT '',
                    members_json TEXT NOT NULL DEFAULT '[]',
                    open_threads_json TEXT NOT NULL DEFAULT '[]',
                    from_ts REAL NOT NULL,
                    to_ts REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    source_kind TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scene_summaries_scope_time
                    ON scene_summaries(scope_id, to_ts DESC);

                CREATE TABLE IF NOT EXISTS memory_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT NOT NULL,
                    original_row_id INTEGER NOT NULL,
                    scope_id TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    kept_row_id INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    archived_at REAL NOT NULL,
                    restored_at REAL NOT NULL DEFAULT 0,
                    restored_row_id INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memory_archive_scope_time
                    ON memory_archive(scope_id, archived_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_archive_table_row
                    ON memory_archive(table_name, original_row_id);

                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );
                """
            )
            self._ensure_column(
                connection,
                "private_memories",
                "memory_key",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "private_memories",
                "topic_key",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                connection,
                "exchanges",
                "memory_key",
                "TEXT NOT NULL DEFAULT ''",
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_private_memory_scope_identity "
                "ON private_memories(scope_id, memory_key, status, pinned, updated_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_private_memory_topic "
                "ON private_memories("
                "scope_id, memory_key, target_id, kind, topic_key, status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_exchanges_scope_identity_time "
                "ON exchanges(scope_id, memory_key, created_at)"
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def adopt_legacy_memory_identity(
        self,
        *,
        scope_id: str,
        bot_id: str,
        memory_key: str,
    ) -> int:
        """Attach pre-memory_key rows to the role currently occupying the account."""
        if not scope_id or not bot_id or not memory_key or memory_key == bot_id:
            return 0
        changed = 0
        with self._connection() as connection:
            for table in ("private_memories", "exchanges"):
                cursor = connection.execute(
                    f"UPDATE {table} SET memory_key = ? "
                    "WHERE scope_id = ? AND bot_id = ? "
                    "AND (memory_key = '' OR memory_key = ?)",
                    (memory_key[:160], scope_id, bot_id, bot_id),
                )
                changed += max(0, int(cursor.rowcount))
        return changed

    def migrate_legacy_records(
        self,
        *,
        scope_groups: dict[str, str],
        bot_ids: dict[str, str],
        memory_keys: dict[str, str],
    ) -> dict[str, Any]:
        """Move legacy platform scopes/account ids into canonical BotMesh identities."""
        backup_path = self.path.with_name(
            f"{self.path.stem}.pre-migrate-{int(time.time() * 1000)}{self.path.suffix}"
        )
        source = self._connect()
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()

        moved_scopes = 0
        moved_identities = 0
        skipped_conflicts = 0
        with self._connection() as connection:
            for table in ("facts", "corrections"):
                rows = connection.execute(
                    f"SELECT id, scope_id FROM {table}"
                ).fetchall()
                for row in rows:
                    group_id = scope_groups.get(str(row["scope_id"]), "")
                    target_scope = f"botmesh:{group_id}" if group_id else ""
                    if not target_scope or target_scope == row["scope_id"]:
                        continue
                    connection.execute(
                        f"UPDATE {table} SET scope_id = ? WHERE id = ?",
                        (target_scope, int(row["id"])),
                    )
                    moved_scopes += 1

            episode_rows = connection.execute(
                "SELECT id, scope_id, episode_key FROM episodes"
            ).fetchall()
            for row in episode_rows:
                group_id = scope_groups.get(str(row["scope_id"]), "")
                target_scope = f"botmesh:{group_id}" if group_id else ""
                if not target_scope or target_scope == row["scope_id"]:
                    continue
                duplicate = connection.execute(
                    "SELECT id FROM episodes WHERE scope_id = ? AND episode_key = ? "
                    "AND id != ? LIMIT 1",
                    (target_scope, str(row["episode_key"]), int(row["id"])),
                ).fetchone()
                if duplicate is not None:
                    skipped_conflicts += 1
                    continue
                connection.execute(
                    "UPDATE episodes SET scope_id = ? WHERE id = ?",
                    (target_scope, int(row["id"])),
                )
                moved_scopes += 1

            for table in ("private_memories", "exchanges"):
                rows = connection.execute(
                    f"SELECT id, scope_id, bot_id, memory_key FROM {table}"
                ).fetchall()
                for row in rows:
                    old_scope = str(row["scope_id"])
                    group_id = scope_groups.get(old_scope, "")
                    if not group_id and old_scope.startswith("botmesh:"):
                        group_id = old_scope[8:]
                    target_scope = f"botmesh:{group_id}" if group_id else old_scope
                    old_bot_id = str(row["bot_id"] or "")
                    target_bot_id = bot_ids.get(old_bot_id, old_bot_id)
                    old_memory_key = str(row["memory_key"] or "")
                    target_memory_key = old_memory_key
                    configured_key = memory_keys.get(
                        f"{group_id}|{target_bot_id}",
                        "",
                    )
                    if configured_key and old_memory_key in (
                        "",
                        old_bot_id,
                        target_bot_id,
                    ):
                        target_memory_key = configured_key
                    if target_scope != old_scope:
                        moved_scopes += 1
                    if (
                        target_bot_id != old_bot_id
                        or target_memory_key != old_memory_key
                    ):
                        moved_identities += 1
                    if (
                        target_scope != old_scope
                        or target_bot_id != old_bot_id
                        or target_memory_key != old_memory_key
                    ):
                        connection.execute(
                            f"UPDATE {table} SET scope_id = ?, bot_id = ?, "
                            "memory_key = ? WHERE id = ?",
                            (
                                target_scope,
                                target_bot_id,
                                target_memory_key[:160],
                                int(row["id"]),
                            ),
                        )
        return {
            "moved_scope_rows": moved_scopes,
            "moved_identity_rows": moved_identities,
            "skipped_conflicts": skipped_conflicts,
            "backup_path": str(backup_path),
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    def record_exchange(
        self,
        *,
        scope_id: str,
        bot_id: str,
        memory_key: str = "",
        umo: str,
        user_message: str,
        assistant_message: str,
        source_kind: str,
        created_at: float | None = None,
    ) -> int:
        now = float(created_at or time.time())
        memory_key = str(memory_key or bot_id).strip()[:160]
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO exchanges(
                    scope_id, bot_id, memory_key, umo, user_message,
                    assistant_message, source_kind, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    bot_id,
                    memory_key[:160],
                    umo,
                    user_message,
                    assistant_message,
                    source_kind,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def upsert_fact(
        self,
        *,
        scope_id: str,
        fact_key: str,
        text: str,
        subject: str = "",
        predicate: str = "",
        object_text: str = "",
        status: str = "inferred",
        confidence: float = 0.5,
        authority: int = 20,
        pinned: bool = False,
        source_kind: str = "",
        source_id: str = "",
        source_excerpt: str = "",
        created_by: str = "",
        merge_target_id: int = 0,
        merge_mode: str = "auto",
        now: float | None = None,
    ) -> int:
        now = float(now or time.time())
        text = str(text or "").strip()[:1000]
        fact_key = str(fact_key or text).strip()[:240]
        status = status if status in {"active", "inferred", "conflict"} else "inferred"
        confidence = max(0.0, min(1.0, float(confidence)))
        authority = max(0, min(100, int(authority)))
        merge_mode = str(merge_mode or "auto").strip().casefold()
        if merge_mode not in {"auto", "append", "replace"}:
            merge_mode = "auto"
        with self._connection() as connection:
            merge_target = None
            if int(merge_target_id or 0) > 0:
                merge_target = connection.execute(
                    """
                    SELECT * FROM facts
                    WHERE id = ? AND scope_id = ?
                      AND status IN ('active', 'inferred', 'conflict')
                    """,
                    (int(merge_target_id), scope_id),
                ).fetchone()
            if merge_target is None and merge_mode == "append":
                merge_target = connection.execute(
                    """
                    SELECT * FROM facts
                    WHERE scope_id = ? AND fact_key = ?
                      AND status IN ('active', 'inferred', 'conflict')
                    ORDER BY pinned DESC, authority DESC, updated_at DESC LIMIT 1
                    """,
                    (scope_id, fact_key),
                ).fetchone()
            if merge_target is not None:
                target_id = int(merge_target["id"])
                if merge_mode == "replace":
                    merged_text = text
                    merged_object = object_text[:500]
                else:
                    merged_text, _ = _merge_memory_text(
                        str(merge_target["text"] or ""),
                        text,
                        max_chars=1000,
                    )
                    merged_object, _ = _merge_memory_text(
                        str(merge_target["object_text"] or ""),
                        object_text,
                        max_chars=500,
                    )
                connection.execute(
                    """
                    UPDATE facts SET
                        text = ?,
                        subject = CASE WHEN subject = '' THEN ? ELSE subject END,
                        predicate = CASE WHEN predicate = '' THEN ? ELSE predicate END,
                        object_text = ?,
                        status = CASE
                            WHEN ? = 'active' THEN 'active' ELSE status
                        END,
                        confidence = MAX(confidence, ?),
                        authority = MAX(authority, ?),
                        pinned = MAX(pinned, ?),
                        source_kind = CASE WHEN ? <> '' THEN ? ELSE source_kind END,
                        source_id = CASE WHEN ? <> '' THEN ? ELSE source_id END,
                        source_excerpt = CASE WHEN ? <> '' THEN ? ELSE source_excerpt END,
                        created_by = CASE WHEN created_by = '' THEN ? ELSE created_by END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_text,
                        subject[:160],
                        predicate[:160],
                        merged_object,
                        status,
                        confidence,
                        authority,
                        int(bool(pinned)),
                        source_kind[:64],
                        source_kind[:64],
                        source_id[:160],
                        source_id[:160],
                        source_excerpt[:1000],
                        source_excerpt[:1000],
                        created_by[:160],
                        now,
                        target_id,
                    ),
                )
                return target_id

            same = connection.execute(
                """
                SELECT id FROM facts
                WHERE scope_id = ? AND fact_key = ? AND text = ?
                  AND status IN ('active', 'inferred', 'conflict')
                ORDER BY id DESC LIMIT 1
                """,
                (scope_id, fact_key, text),
            ).fetchone()
            if same is not None:
                connection.execute(
                    """
                    UPDATE facts SET
                        confidence = MAX(confidence, ?),
                        authority = MAX(authority, ?),
                        pinned = MAX(pinned, ?),
                        status = CASE
                            WHEN status = 'active' OR ? = 'active' THEN 'active'
                            ELSE status
                        END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        confidence,
                        authority,
                        int(bool(pinned)),
                        status,
                        now,
                        int(same["id"]),
                    ),
                )
                return int(same["id"])

            current = connection.execute(
                """
                SELECT id, authority, text FROM facts
                WHERE scope_id = ? AND fact_key = ? AND status = 'active'
                ORDER BY pinned DESC, authority DESC, updated_at DESC LIMIT 1
                """,
                (scope_id, fact_key),
            ).fetchone()
            if current is not None and str(current["text"]) != text:
                if status == "active" and authority >= int(current["authority"]):
                    connection.execute(
                        "UPDATE facts SET status = 'superseded', updated_at = ? WHERE id = ?",
                        (now, int(current["id"])),
                    )
                elif status == "active":
                    status = "conflict"

            cursor = connection.execute(
                """
                INSERT INTO facts(
                    scope_id, fact_key, text, subject, predicate, object_text,
                    status, confidence, authority, pinned, source_kind,
                    source_id, source_excerpt, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    fact_key,
                    text,
                    subject[:160],
                    predicate[:160],
                    object_text[:500],
                    status,
                    confidence,
                    authority,
                    int(bool(pinned)),
                    source_kind[:64],
                    source_id[:160],
                    source_excerpt[:1000],
                    created_by[:160],
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def add_private_memory(
        self,
        *,
        scope_id: str,
        bot_id: str,
        memory_key: str = "",
        target_id: str,
        kind: str,
        topic_key: str = "",
        text: str,
        confidence: float,
        source_kind: str,
        source_id: str,
        pinned: bool = False,
        merge_target_id: int = 0,
        merge_mode: str = "auto",
        now: float | None = None,
    ) -> int:
        now = float(now or time.time())
        memory_key = str(memory_key or bot_id).strip()[:160]
        text = str(text or "").strip()[:1000]
        target_id = str(target_id or "").strip()[:160]
        kind = str(kind or "impression").strip()[:64]
        topic_key = str(topic_key or "").strip()[:240]
        confidence = max(0.0, min(1.0, float(confidence)))
        merge_mode = str(merge_mode or "auto").strip().casefold()
        if merge_mode not in {"auto", "append", "replace"}:
            merge_mode = "auto"
        with self._connection() as connection:
            merge_target = None
            if int(merge_target_id or 0) > 0:
                merge_target = connection.execute(
                    """
                    SELECT * FROM private_memories
                    WHERE id = ? AND scope_id = ? AND memory_key = ?
                      AND target_id = ? AND kind = ? AND status = 'active'
                    """,
                    (
                        int(merge_target_id),
                        scope_id,
                        memory_key,
                        target_id,
                        kind,
                    ),
                ).fetchone()
            if merge_target is None and topic_key:
                merge_target = connection.execute(
                    """
                    SELECT * FROM private_memories
                    WHERE scope_id = ? AND memory_key = ? AND target_id = ?
                      AND kind = ? AND topic_key = ? AND status = 'active'
                    ORDER BY pinned DESC, updated_at DESC LIMIT 1
                    """,
                    (scope_id, memory_key, target_id, kind, topic_key),
                ).fetchone()
            if merge_target is not None:
                target_row_id = int(merge_target["id"])
                effective_mode = merge_mode
                if effective_mode == "auto":
                    effective_mode = "replace" if kind == "emotion" else "append"
                if effective_mode == "replace":
                    merged_text = text
                else:
                    merged_text, _ = _merge_memory_text(
                        str(merge_target["text"] or ""),
                        text,
                        max_chars=1000,
                    )
                connection.execute(
                    """
                    UPDATE private_memories SET
                        bot_id = ?, text = ?,
                        topic_key = CASE WHEN topic_key = '' THEN ? ELSE topic_key END,
                        confidence = MAX(confidence, ?),
                        pinned = MAX(pinned, ?),
                        source_kind = CASE WHEN ? <> '' THEN ? ELSE source_kind END,
                        source_id = CASE WHEN ? <> '' THEN ? ELSE source_id END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        bot_id,
                        merged_text,
                        topic_key,
                        confidence,
                        int(bool(pinned)),
                        source_kind[:64],
                        source_kind[:64],
                        source_id[:160],
                        source_id[:160],
                        now,
                        target_row_id,
                    ),
                )
                return target_row_id

            same = connection.execute(
                """
                SELECT id FROM private_memories
                WHERE scope_id = ? AND memory_key = ? AND target_id = ?
                  AND kind = ? AND text = ? AND status = 'active'
                ORDER BY id DESC LIMIT 1
                """,
                (scope_id, memory_key, target_id, kind, text),
            ).fetchone()
            if same is not None:
                connection.execute(
                    "UPDATE private_memories SET confidence = MAX(confidence, ?), "
                    "pinned = MAX(pinned, ?), updated_at = ? WHERE id = ?",
                    (
                        confidence,
                        int(bool(pinned)),
                        now,
                        int(same["id"]),
                    ),
                )
                return int(same["id"])
            cursor = connection.execute(
                """
                INSERT INTO private_memories(
                    scope_id, bot_id, memory_key, target_id, kind, topic_key,
                    text, confidence, pinned, status, source_kind, source_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    bot_id,
                    memory_key[:160],
                    target_id[:160],
                    kind[:64],
                    topic_key,
                    text,
                    confidence,
                    int(bool(pinned)),
                    source_kind[:64],
                    source_id[:160],
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def upsert_episode(
        self,
        *,
        scope_id: str,
        episode_key: str,
        title: str,
        summary: str,
        participants: list[str],
        confirmed: list[str],
        unresolved: list[str],
        source_kind: str,
        source_id: str,
        started_at: float,
        ended_at: float,
        merge_target_id: int = 0,
        merge_mode: str = "append",
        now: float | None = None,
    ) -> int:
        now = float(now or time.time())
        merge_mode = str(merge_mode or "append").strip().casefold()
        if merge_mode not in {"append", "replace"}:
            merge_mode = "append"
        with self._connection() as connection:
            row = None
            if int(merge_target_id or 0) > 0:
                row = connection.execute(
                    "SELECT * FROM episodes WHERE id = ? AND scope_id = ?",
                    (int(merge_target_id), scope_id),
                ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM episodes WHERE scope_id = ? AND episode_key = ?",
                    (scope_id, episode_key),
                ).fetchone()
            if row is not None:
                if merge_mode == "replace":
                    merged_title = title[:240]
                    merged_summary = summary[:2000]
                    merged_participants = _merge_string_lists(participants, limit=30)
                    merged_confirmed = _merge_string_lists(confirmed, limit=30)
                    merged_unresolved = _merge_string_lists(unresolved, limit=30)
                else:
                    merged_title = str(row["title"] or "").strip() or title[:240]
                    merged_summary, _ = _merge_memory_text(
                        str(row["summary"] or ""),
                        summary,
                        max_chars=2000,
                    )
                    merged_participants = _merge_string_lists(
                        row["participants_json"], participants, limit=30
                    )
                    merged_confirmed = _merge_string_lists(
                        row["confirmed_json"], confirmed, limit=30
                    )
                    merged_unresolved = _merge_string_lists(
                        row["unresolved_json"], unresolved, limit=30
                    )
                    confirmed_keys = {
                        _normalized_text(item) for item in merged_confirmed
                    }
                    merged_unresolved = [
                        item
                        for item in merged_unresolved
                        if _normalized_text(item) not in confirmed_keys
                    ]
                connection.execute(
                    """
                    UPDATE episodes SET title = ?, summary = ?, participants_json = ?,
                        confirmed_json = ?, unresolved_json = ?, source_kind = ?,
                        source_id = ?, started_at = MIN(started_at, ?),
                        ended_at = MAX(ended_at, ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_title,
                        merged_summary,
                        json.dumps(merged_participants, ensure_ascii=False),
                        json.dumps(merged_confirmed, ensure_ascii=False),
                        json.dumps(merged_unresolved, ensure_ascii=False),
                        source_kind[:64],
                        source_id[:160],
                        float(started_at),
                        float(ended_at),
                        now,
                        int(row["id"]),
                    ),
                )
                return int(row["id"])
            values = (
                title[:240],
                summary[:2000],
                json.dumps(_merge_string_lists(participants, limit=30), ensure_ascii=False),
                json.dumps(_merge_string_lists(confirmed, limit=30), ensure_ascii=False),
                json.dumps(_merge_string_lists(unresolved, limit=30), ensure_ascii=False),
                source_kind[:64],
                source_id[:160],
                float(started_at),
                float(ended_at),
                now,
            )
            cursor = connection.execute(
                """
                INSERT INTO episodes(
                    scope_id, episode_key, title, summary, participants_json,
                    confirmed_json, unresolved_json, status, source_kind,
                    source_id, started_at, ended_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (scope_id, episode_key, *values[:-1], now, now),
            )
            return int(cursor.lastrowid)

    def add_correction(
        self,
        *,
        scope_id: str,
        old_text: str,
        new_text: str,
        reason: str,
        source_kind: str,
        source_id: str,
        now: float | None = None,
    ) -> int:
        now = float(now or time.time())
        with self._connection() as connection:
            old_text = str(old_text or "").strip()[:1000]
            new_text = str(new_text or "").strip()[:1000]
            if old_text:
                connection.execute(
                    "UPDATE facts SET status = 'superseded', updated_at = ? "
                    "WHERE scope_id = ? AND text = ? "
                    "AND status IN ('active', 'inferred', 'conflict')",
                    (now, scope_id, old_text),
                )
            cursor = connection.execute(
                """
                INSERT INTO corrections(
                    scope_id, old_text, new_text, reason,
                    source_kind, source_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    old_text,
                    new_text,
                    reason[:500],
                    source_kind[:64],
                    source_id[:160],
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def retrieve(
        self,
        *,
        scope_id: str,
        bot_id: str,
        memory_key: str = "",
        fact_limit: int,
        private_limit: int,
        episode_limit: int,
        query: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        memory_key = str(memory_key or bot_id).strip()[:160]
        tokens = _query_tokens(query)
        candidate_multiplier = 10 if tokens else 3
        with self._connection() as connection:
            facts = connection.execute(
                """
                SELECT * FROM facts
                WHERE scope_id = ? AND status IN ('active', 'inferred', 'conflict')
                ORDER BY pinned DESC,
                    CASE status WHEN 'active' THEN 0 WHEN 'conflict' THEN 1 ELSE 2 END,
                    authority DESC, confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (
                    scope_id,
                    min(500, max(1, int(fact_limit)) * candidate_multiplier),
                ),
            ).fetchall()
            private = connection.execute(
                """
                SELECT * FROM private_memories
                WHERE scope_id = ? AND memory_key = ? AND status = 'active'
                ORDER BY pinned DESC, confidence DESC, updated_at DESC LIMIT ?
                """,
                (
                    scope_id,
                    memory_key,
                    min(500, max(1, int(private_limit)) * candidate_multiplier),
                ),
            ).fetchall()
            episodes = connection.execute(
                """
                SELECT * FROM episodes
                WHERE scope_id = ? AND status = 'active'
                ORDER BY ended_at DESC LIMIT ?
                """,
                (
                    scope_id,
                    min(500, max(1, int(episode_limit)) * candidate_multiplier),
                ),
            ).fetchall()
            corrections = connection.execute(
                """
                SELECT * FROM corrections WHERE scope_id = ?
                ORDER BY created_at DESC LIMIT 20
                """,
                (scope_id,),
            ).fetchall()
            exchanges = connection.execute(
                """
                SELECT * FROM exchanges
                WHERE scope_id = ? AND memory_key = ?
                ORDER BY created_at DESC LIMIT 12
                """,
                (scope_id, memory_key),
            ).fetchall()
        status_rank = {"active": 0, "conflict": 1}
        fact_limit = max(1, int(fact_limit))
        private_limit = max(1, int(private_limit))
        episode_limit = max(1, int(episode_limit))
        facts = sorted(
            facts,
            key=lambda row: (
                0 if int(row["pinned"] or 0) else 1,
                -_text_relevance(
                    " ".join(
                        str(row[field] or "")
                        for field in ("text", "subject", "predicate", "object_text")
                    ),
                    tokens,
                ),
                status_rank.get(str(row["status"] or ""), 2),
                -int(row["authority"] or 0),
                -float(row["confidence"] or 0),
                -float(row["updated_at"] or 0),
            ),
        )
        facts = _retain_relevant(
            facts,
            fields=("text", "subject", "predicate", "object_text"),
            tokens=tokens,
            fallback=min(fact_limit, 8),
        )
        facts = _dedupe_similar(
            facts,
            fields=("text",),
            limit=fact_limit,
            threshold=0.4,
        )
        private = sorted(
            private,
            key=lambda row: (
                0 if int(row["pinned"] or 0) else 1,
                -_text_relevance(
                    " ".join(
                        str(row[field] or "")
                        for field in ("text", "kind")
                    ),
                    tokens,
                ),
                -float(row["confidence"] or 0),
                -float(row["updated_at"] or 0),
            ),
        )
        private = _retain_relevant(
            private,
            fields=("text", "kind", "target_id"),
            tokens=tokens,
            fallback=min(private_limit, 6),
        )
        private = _limit_per_group(
            private,
            fields=("target_id", "kind"),
            per_group=4,
        )
        private = _dedupe_similar(
            private,
            fields=("text",),
            limit=private_limit,
            threshold=0.45,
        )
        episodes = sorted(
            episodes,
            key=lambda row: (
                -_text_relevance(
                    " ".join(
                        str(row[field] or "")
                        for field in ("title", "summary")
                    ),
                    tokens,
                ),
                -float(row["ended_at"] or 0),
            ),
        )
        episodes = _retain_relevant(
            episodes,
            fields=("title", "summary"),
            tokens=tokens,
            fallback=min(episode_limit, 2),
        )
        episodes = _dedupe_similar(
            episodes,
            fields=("title", "summary"),
            limit=episode_limit,
            threshold=0.35,
        )
        corrections = sorted(
            corrections,
            key=lambda row: (
                -_text_relevance(
                    f"{row['old_text'] or ''} {row['new_text'] or ''}",
                    tokens,
                ),
                -float(row["created_at"] or 0),
            ),
        )
        corrections = _retain_relevant(
            corrections,
            fields=("old_text", "new_text", "reason"),
            tokens=tokens,
            fallback=2,
        )
        corrections = _dedupe_similar(
            corrections,
            fields=("old_text", "new_text"),
            limit=4,
            threshold=0.4,
        )
        exchanges = sorted(
            exchanges,
            key=lambda row: (
                -_text_relevance(
                    f"{row['user_message'] or ''} {row['assistant_message'] or ''}",
                    tokens,
                ),
                -float(row["created_at"] or 0),
            ),
        )
        exchanges = _retain_relevant(
            exchanges,
            fields=("user_message", "assistant_message"),
            tokens=tokens,
            fallback=2,
        )
        exchanges = _dedupe_similar(
            exchanges,
            fields=("assistant_message",),
            limit=4,
            threshold=0.45,
        )
        return {
            "facts": [self._row(row) for row in facts],
            "private_memories": [self._row(row) for row in private],
            "episodes": [self._row(row) for row in episodes],
            "corrections": [self._row(row) for row in corrections],
            "self_exchanges": [self._row(row) for row in exchanges],
        }

    def collapse_duplicate_facts(self, scope_id: str = "") -> dict[str, int]:
        """合并数据库里高度相似的事实：保留最佳一条，其余标记 superseded。

        只处理 active/inferred/conflict 状态；scope_id 为空时扫描全部群。
        返回 {"scanned", "kept", "superseded"}。
        """
        clauses = ["status IN ('active', 'inferred', 'conflict')"]
        params: list[Any] = []
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(str(scope_id).strip())
        sql = (
            "SELECT * FROM facts WHERE "
            + " AND ".join(clauses)
            + " ORDER BY pinned DESC, "
            + "CASE status WHEN 'active' THEN 0 WHEN 'conflict' THEN 1 ELSE 2 END, "
            + "authority DESC, confidence DESC, updated_at DESC"
        )
        kept: list[sqlite3.Row] = []
        superseded: list[tuple[int, int]] = []
        now = time.time()
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
            for row in rows:
                text = str(row["text"] or "")
                duplicate_id: int | None = None
                for previous in kept:
                    if str(previous["scope_id"] or "") != str(row["scope_id"] or ""):
                        continue
                    if _text_similarity(text, str(previous["text"] or "")) >= 0.68:
                        duplicate_id = int(previous["id"])
                        break
                if duplicate_id is None:
                    kept.append(row)
                else:
                    superseded.append((int(row["id"]), duplicate_id))
            for fact_id, kept_id in superseded:
                connection.execute(
                    "UPDATE facts SET status = 'superseded', updated_at = ? "
                    "WHERE id = ?",
                    (now, fact_id),
                )
        return {
            "scanned": len(rows),
            "kept": len(kept),
            "superseded": len(superseded),
        }

    @staticmethod
    def _normalized_memory_text(value: Any) -> str:
        return re.sub(r"[\W_]+", "", str(value or ""), flags=re.UNICODE).casefold()

    @staticmethod
    def _archive_row(
        connection: sqlite3.Connection,
        *,
        table_name: str,
        row: sqlite3.Row,
        action: str,
        archived_at: float,
        kept_row_id: int = 0,
    ) -> int:
        payload = dict(row)
        cursor = connection.execute(
            """
            INSERT INTO memory_archive(
                table_name, original_row_id, scope_id, action, kept_row_id,
                payload_json, archived_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table_name,
                int(payload.get("id") or 0),
                str(payload.get("scope_id") or ""),
                str(action or "maintenance")[:64],
                max(0, int(kept_row_id or 0)),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                float(archived_at),
            ),
        )
        return int(cursor.lastrowid or 0)

    @staticmethod
    def _memory_rows_are_duplicates(
        first: sqlite3.Row,
        second: sqlite3.Row,
        *,
        kind: str,
    ) -> bool:
        first_text = str(first["text"] or "")
        second_text = str(second["text"] or "")
        first_normalized = MemoryStore._normalized_memory_text(first_text)
        second_normalized = MemoryStore._normalized_memory_text(second_text)
        if first_normalized and first_normalized == second_normalized:
            return True
        similarity = _text_similarity(first_text, second_text)
        if kind == "fact":
            same_key = str(first["fact_key"] or "") == str(second["fact_key"] or "")
            return similarity >= (0.88 if same_key else 0.94)
        return similarity >= 0.94

    def _duplicate_pairs(
        self,
        connection: sqlite3.Connection,
        *,
        scope_id: str,
        max_rows: int,
    ) -> list[tuple[str, sqlite3.Row, sqlite3.Row]]:
        """Return (table, duplicate, keeper) pairs using conservative boundaries."""
        target = str(scope_id or "").strip()
        where = " AND scope_id = ?" if target else ""
        params: tuple[Any, ...] = (target,) if target else ()
        fact_rows = connection.execute(
            "SELECT * FROM facts WHERE status IN ('active','inferred','conflict')"
            + where
            + " ORDER BY scope_id, pinned DESC, "
            + "CASE status WHEN 'active' THEN 0 WHEN 'conflict' THEN 1 ELSE 2 END, "
            + "authority DESC, confidence DESC, updated_at DESC LIMIT ?",
            (*params, max(50, int(max_rows) * 5)),
        ).fetchall()
        private_rows = connection.execute(
            "SELECT * FROM private_memories WHERE status = 'active'"
            + where
            + " ORDER BY scope_id, memory_key, pinned DESC, confidence DESC, "
            + "updated_at DESC LIMIT ?",
            (*params, max(50, int(max_rows) * 5)),
        ).fetchall()

        pairs: list[tuple[str, sqlite3.Row, sqlite3.Row]] = []
        fact_kept: dict[str, list[sqlite3.Row]] = {}
        for row in fact_rows:
            group = str(row["scope_id"] or "")
            keepers = fact_kept.setdefault(group, [])
            keeper = next(
                (
                    item
                    for item in keepers
                    if self._memory_rows_are_duplicates(item, row, kind="fact")
                ),
                None,
            )
            if keeper is None:
                keepers.append(row)
            else:
                pairs.append(("facts", row, keeper))
                if len(pairs) >= max_rows:
                    return pairs

        private_kept: dict[tuple[str, str, str, str], list[sqlite3.Row]] = {}
        for row in private_rows:
            group = (
                str(row["scope_id"] or ""),
                str(row["memory_key"] or ""),
                str(row["target_id"] or ""),
                str(row["kind"] or ""),
            )
            keepers = private_kept.setdefault(group, [])
            keeper = next(
                (
                    item
                    for item in keepers
                    if self._memory_rows_are_duplicates(item, row, kind="private")
                ),
                None,
            )
            if keeper is None:
                keepers.append(row)
            else:
                pairs.append(("private_memories", row, keeper))
                if len(pairs) >= max_rows:
                    break
        return pairs

    def maintenance_status(self, *, scope_id: str = "") -> dict[str, Any]:
        target = str(scope_id or "").strip()
        state_key = "maintenance:last_run" + (f":{target}" if target else "")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value_json, updated_at FROM memory_meta "
                "WHERE key = ?",
                (state_key,),
            ).fetchone()
            if target:
                archive_row = connection.execute(
                    "SELECT COUNT(*) FROM memory_archive "
                    "WHERE restored_at = 0 AND scope_id = ?",
                    (target,),
                ).fetchone()
            else:
                archive_row = connection.execute(
                    "SELECT COUNT(*) FROM memory_archive WHERE restored_at = 0"
                ).fetchone()
            archive_count = int(archive_row[0] or 0)
        payload: dict[str, Any] = {}
        if row is not None:
            try:
                parsed = json.loads(str(row["value_json"] or "{}"))
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = {}
            payload["updated_at"] = float(row["updated_at"] or 0)
        payload["archive_count"] = archive_count
        return payload

    def maintain(
        self,
        *,
        scope_id: str = "",
        now: float | None = None,
        dry_run: bool = False,
        dedupe_enabled: bool = True,
        fact_ttl_days: int = 365,
        superseded_fact_ttl_days: int = 90,
        protected_fact_authority: int = 80,
        private_memory_ttl_days: int = 365,
        episode_ttl_days: int = 180,
        correction_ttl_days: int = 365,
        exchange_ttl_days: int = 30,
        scene_ttl_days: int = 30,
        archive_ttl_days: int = 365,
        max_rows: int = 1000,
    ) -> dict[str, Any]:
        """Archive, compact and expire memory rows in one transaction.

        TTL values <= 0 disable expiry for that category. Pinned rows and active
        facts at or above ``protected_fact_authority`` never expire automatically.
        """
        ran_at = float(now or time.time())
        target = str(scope_id or "").strip()
        row_budget = max(10, min(int(max_rows or 1000), 10000))
        stats: dict[str, Any] = {
            "ran_at": ran_at,
            "scope_id": target,
            "dry_run": bool(dry_run),
            "deduplicated": {"facts": 0, "private_memories": 0},
            "expired": {
                "facts": 0,
                "private_memories": 0,
                "episodes": 0,
                "corrections": 0,
                "exchanges": 0,
                "scene_summaries": 0,
            },
            "archives_purged": 0,
            "archived": 0,
        }
        with self._connection() as connection:
            duplicate_ids: dict[str, set[int]] = {
                "facts": set(),
                "private_memories": set(),
            }
            if dedupe_enabled:
                pairs = self._duplicate_pairs(
                    connection,
                    scope_id=target,
                    max_rows=row_budget,
                )
                for table, duplicate, keeper in pairs:
                    duplicate_id = int(duplicate["id"])
                    keeper_id = int(keeper["id"])
                    duplicate_ids[table].add(duplicate_id)
                    stats["deduplicated"][table] += 1
                    if dry_run:
                        continue
                    self._archive_row(
                        connection,
                        table_name=table,
                        row=duplicate,
                        action="deduplicated",
                        archived_at=ran_at,
                        kept_row_id=keeper_id,
                    )
                    if table == "facts":
                        connection.execute(
                            "UPDATE facts SET confidence = MAX(confidence, ?), "
                            "authority = MAX(authority, ?), pinned = MAX(pinned, ?), "
                            "created_at = MIN(created_at, ?), "
                            "updated_at = MAX(updated_at, ?) WHERE id = ?",
                            (
                                float(duplicate["confidence"] or 0),
                                int(duplicate["authority"] or 0),
                                int(duplicate["pinned"] or 0),
                                float(duplicate["created_at"] or ran_at),
                                float(duplicate["updated_at"] or ran_at),
                                keeper_id,
                            ),
                        )
                    else:
                        connection.execute(
                            "UPDATE private_memories SET "
                            "confidence = MAX(confidence, ?), pinned = MAX(pinned, ?), "
                            "created_at = MIN(created_at, ?), "
                            "updated_at = MAX(updated_at, ?) WHERE id = ?",
                            (
                                float(duplicate["confidence"] or 0),
                                int(duplicate["pinned"] or 0),
                                float(duplicate["created_at"] or ran_at),
                                float(duplicate["updated_at"] or ran_at),
                                keeper_id,
                            ),
                        )
                    connection.execute(
                        f"DELETE FROM {table} WHERE id = ?", (duplicate_id,)
                    )
                    stats["archived"] += 1

            remaining = max(0, row_budget - sum(stats["deduplicated"].values()))
            expiry_queries: list[tuple[str, str, tuple[Any, ...]]] = []
            scope_sql = " AND scope_id = ?" if target else ""
            scope_params: tuple[Any, ...] = (target,) if target else ()

            fact_parts: list[str] = []
            fact_params: list[Any] = []
            if int(superseded_fact_ttl_days) > 0:
                fact_parts.append("(status = 'superseded' AND updated_at < ?)")
                fact_params.append(ran_at - int(superseded_fact_ttl_days) * 86400)
            if int(fact_ttl_days) > 0:
                fact_parts.append(
                    "(status != 'superseded' AND updated_at < ? "
                    "AND NOT (status = 'active' AND authority >= ?))"
                )
                fact_params.extend(
                    [
                        ran_at - int(fact_ttl_days) * 86400,
                        max(0, min(100, int(protected_fact_authority))),
                    ]
                )
            if fact_parts:
                expiry_queries.append(
                    (
                        "facts",
                        "pinned = 0 AND (" + " OR ".join(fact_parts) + ")" + scope_sql,
                        (*fact_params, *scope_params),
                    )
                )

            simple_policies = (
                ("private_memories", "updated_at", int(private_memory_ttl_days), True),
                ("episodes", "ended_at", int(episode_ttl_days), False),
                ("corrections", "created_at", int(correction_ttl_days), False),
                ("exchanges", "created_at", int(exchange_ttl_days), False),
                ("scene_summaries", "to_ts", int(scene_ttl_days), False),
            )
            for table, timestamp_column, ttl_days, has_pinned in simple_policies:
                if ttl_days <= 0:
                    continue
                condition = f"{timestamp_column} < ?"
                if has_pinned:
                    condition = "pinned = 0 AND " + condition
                expiry_queries.append(
                    (
                        table,
                        condition + scope_sql,
                        (ran_at - ttl_days * 86400, *scope_params),
                    )
                )

            for table, condition, params in expiry_queries:
                if remaining <= 0:
                    break
                rows = connection.execute(
                    f"SELECT * FROM {table} WHERE {condition} "
                    "ORDER BY id LIMIT ?",
                    (*params, remaining),
                ).fetchall()
                excluded = duplicate_ids.get(table, set())
                rows = [row for row in rows if int(row["id"]) not in excluded]
                stats["expired"][table] += len(rows)
                remaining -= len(rows)
                if dry_run:
                    continue
                for row in rows:
                    self._archive_row(
                        connection,
                        table_name=table,
                        row=row,
                        action="expired",
                        archived_at=ran_at,
                    )
                    connection.execute(
                        f"DELETE FROM {table} WHERE id = ?", (int(row["id"]),)
                    )
                    stats["archived"] += 1

            if int(archive_ttl_days) > 0:
                archive_cutoff = ran_at - int(archive_ttl_days) * 86400
                if target:
                    archived_rows = connection.execute(
                        "SELECT id FROM memory_archive WHERE archived_at < ? "
                        "AND scope_id = ? ORDER BY id LIMIT ?",
                        (archive_cutoff, target, row_budget),
                    ).fetchall()
                else:
                    archived_rows = connection.execute(
                        "SELECT id FROM memory_archive WHERE archived_at < ? "
                        "ORDER BY id LIMIT ?",
                        (archive_cutoff, row_budget),
                    ).fetchall()
                stats["archives_purged"] = len(archived_rows)
                if archived_rows and not dry_run:
                    placeholders = ",".join("?" for _ in archived_rows)
                    connection.execute(
                        f"DELETE FROM memory_archive WHERE id IN ({placeholders})",
                        tuple(int(row["id"]) for row in archived_rows),
                    )

            stats["would_archive"] = (
                sum(stats["deduplicated"].values())
                + sum(stats["expired"].values())
            )
            stats["total_changed"] = (
                sum(stats["deduplicated"].values())
                + sum(stats["expired"].values())
                + int(stats["archives_purged"])
            )
            if not dry_run:
                state_key = "maintenance:last_run" + (
                    f":{target}" if target else ""
                )
                connection.execute(
                    "INSERT INTO memory_meta(key, value_json, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value_json = excluded.value_json, updated_at = excluded.updated_at",
                    (
                        state_key,
                        json.dumps(stats, ensure_ascii=False, separators=(",", ":")),
                        ran_at,
                    ),
                )
        return stats

    def list_archive(
        self,
        *,
        scope_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        target = str(scope_id or "").strip()
        where = "WHERE restored_at = 0"
        params: list[Any] = []
        if target:
            where += " AND scope_id = ?"
            params.append(target)
        params.append(max(1, min(int(limit or 20), 100)))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, table_name, original_row_id, scope_id, action, "
                "kept_row_id, archived_at FROM memory_archive "
                f"{where} ORDER BY archived_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row(row) for row in rows]

    def restore_archive(self, archive_id: int, *, scope_id: str = "") -> dict[str, Any]:
        allowed_tables = {
            "facts",
            "private_memories",
            "episodes",
            "corrections",
            "exchanges",
            "scene_summaries",
        }
        target = str(scope_id or "").strip()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memory_archive WHERE id = ? AND restored_at = 0",
                (int(archive_id),),
            ).fetchone()
            if row is None:
                return {"success": False, "error": "not_found"}
            if target and str(row["scope_id"] or "") != target:
                return {"success": False, "error": "scope_mismatch"}
            table = str(row["table_name"] or "")
            if table not in allowed_tables:
                return {"success": False, "error": "unsupported_table"}
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError):
                return {"success": False, "error": "invalid_payload"}
            if not isinstance(payload, dict):
                return {"success": False, "error": "invalid_payload"}
            columns = {
                str(item["name"])
                for item in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            values = {key: value for key, value in payload.items() if key in columns}
            original_id = int(values.get("id") or 0)
            if original_id and connection.execute(
                f"SELECT 1 FROM {table} WHERE id = ?", (original_id,)
            ).fetchone() is not None:
                values.pop("id", None)
            if not values:
                return {"success": False, "error": "empty_payload"}
            names = list(values)
            try:
                cursor = connection.execute(
                    f"INSERT INTO {table} ({','.join(names)}) VALUES "
                    f"({','.join('?' for _ in names)})",
                    tuple(values[name] for name in names),
                )
            except sqlite3.IntegrityError as exc:
                return {"success": False, "error": "conflict", "detail": str(exc)}
            restored_row_id = int(cursor.lastrowid or original_id)
            restored_at = time.time()
            connection.execute(
                "UPDATE memory_archive SET restored_at = ?, restored_row_id = ? "
                "WHERE id = ?",
                (restored_at, restored_row_id, int(archive_id)),
            )
        return {
            "success": True,
            "archive_id": int(archive_id),
            "table_name": table,
            "row_id": restored_row_id,
        }

    def latest_scene(self, scope_id: str) -> dict[str, Any] | None:
        scope_id = str(scope_id or "").strip()
        if not scope_id:
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM scene_summaries WHERE scope_id = ? "
                "ORDER BY to_ts DESC, id DESC LIMIT 1",
                (scope_id,),
            ).fetchone()
        return self._row(row) if row is not None else None

    def save_scene(
        self,
        scope_id: str,
        *,
        title: str = "",
        summary: str = "",
        topic: str = "",
        mood: str = "",
        members: list[str] | None = None,
        open_threads: list[str] | None = None,
        from_ts: float,
        to_ts: float,
        message_count: int = 0,
        source_kind: str = "scene_summary",
    ) -> int:
        now = time.time()
        scope_id = str(scope_id or "").strip()
        if not scope_id:
            return 0
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO scene_summaries(
                    scope_id, title, summary, topic, mood,
                    members_json, open_threads_json,
                    from_ts, to_ts, message_count, source_kind,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    str(title or "")[:200],
                    str(summary or "")[:8000],
                    str(topic or "")[:1000],
                    str(mood or "")[:500],
                    json.dumps(members or [], ensure_ascii=False)[:4000],
                    json.dumps(open_threads or [], ensure_ascii=False)[:4000],
                    float(from_ts),
                    float(to_ts),
                    max(0, int(message_count or 0)),
                    str(source_kind or "scene_summary")[:64],
                    now,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM scene_summaries WHERE id IN ("
                "SELECT id FROM scene_summaries WHERE scope_id = ? "
                "ORDER BY to_ts DESC, id DESC LIMIT -1 OFFSET 16)",
                (scope_id,),
            )
            return int(cursor.lastrowid or 0)

    def list_scenes(self, scope_id: str, limit: int = 8) -> list[dict[str, Any]]:
        scope_id = str(scope_id or "").strip()
        if not scope_id:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM scene_summaries WHERE scope_id = ? "
                "ORDER BY to_ts DESC, id DESC LIMIT ?",
                (scope_id, max(1, min(int(limit), 50))),
            ).fetchall()
        return [self._row(row) for row in rows]

    def workspace(self, scope_id: str = "") -> dict[str, list[dict[str, Any]]]:
        where = " WHERE scope_id = ?" if scope_id else ""
        params: tuple[object, ...] = (scope_id,) if scope_id else ()
        schedule_where = (
            " WHERE scope_id = ? AND source_kind = 'dynamic_life_state'"
            if scope_id
            else " WHERE source_kind = 'dynamic_life_state'"
        )
        with self._connection() as connection:
            facts = connection.execute(
                f"SELECT * FROM facts{where} ORDER BY created_at DESC, id DESC LIMIT 500",
                params,
            ).fetchall()
            private = connection.execute(
                f"SELECT * FROM private_memories{where} "
                "ORDER BY created_at DESC, id DESC LIMIT 500",
                params,
            ).fetchall()
            episodes = connection.execute(
                f"SELECT * FROM episodes{where} ORDER BY ended_at DESC LIMIT 300",
                params,
            ).fetchall()
            corrections = connection.execute(
                f"SELECT * FROM corrections{where} ORDER BY created_at DESC LIMIT 300",
                params,
            ).fetchall()
            schedules = connection.execute(
                "SELECT id, scope_id, bot_id, memory_key, assistant_message, "
                "source_kind, created_at FROM exchanges"
                f"{schedule_where} ORDER BY created_at DESC, id DESC LIMIT 500",
                params,
            ).fetchall()
            scopes = connection.execute(
                """
                SELECT scope_id, MAX(updated_at) updated_at, COUNT(*) fact_count
                FROM facts GROUP BY scope_id ORDER BY updated_at DESC
                """
            ).fetchall()
        return {
            "facts": [self._row(row) for row in facts],
            "private_memories": [self._row(row) for row in private],
            "episodes": [self._row(row) for row in episodes],
            "corrections": [self._row(row) for row in corrections],
            "schedules": [self._row(row) for row in schedules],
            "scopes": [self._row(row) for row in scopes],
        }

    def update_item(
        self,
        *,
        kind: str,
        item_id: int,
        values: dict[str, Any],
        scope_id: str = "",
    ) -> bool:
        definitions = {
            "fact": (
                "facts",
                {"text", "status", "confidence", "authority", "pinned"},
            ),
            "private": (
                "private_memories",
                {"text", "status", "confidence", "pinned", "kind", "target_id"},
            ),
            "episode": ("episodes", {"title", "summary", "status"}),
        }
        if kind not in definitions:
            return False
        table, allowed = definitions[kind]
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        sql = ", ".join(f"{key} = ?" for key in updates)
        where = "id = ?"
        params: list[Any] = [*updates.values(), int(item_id)]
        if str(scope_id or "").strip():
            where += " AND scope_id = ?"
            params.append(str(scope_id).strip())
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE {table} SET {sql} WHERE {where}",
                params,
            )
            return int(cursor.rowcount) > 0

    def forget(self, *, kind: str, item_id: int, scope_id: str = "") -> bool:
        tables = {
            "fact": "facts",
            "private": "private_memories",
            "episode": "episodes",
            "correction": "corrections",
        }
        table = tables.get(kind)
        if not table:
            return False
        where = "id = ?"
        params: list[Any] = [int(item_id)]
        if str(scope_id or "").strip():
            where += " AND scope_id = ?"
            params.append(str(scope_id).strip())
        with self._connection() as connection:
            cursor = connection.execute(
                f"DELETE FROM {table} WHERE {where}",
                params,
            )
            return int(cursor.rowcount) > 0

    def scope_counts(self, *, scope_id: str) -> dict[str, int]:
        """统计一个记忆作用域内各表的数据条数。"""
        tables = ("facts", "private_memories", "episodes", "corrections", "exchanges")
        result: dict[str, int] = {}
        target = str(scope_id or "").strip()
        with self._connection() as connection:
            for table in tables:
                row = connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE scope_id = ?",
                    (target,),
                ).fetchone()
                result[table] = int(row[0] or 0)
        return result

    def clear_scope(self, *, scope_id: str) -> dict[str, int]:
        """删除一个记忆作用域内的全部记忆数据，返回各表删除条数。"""
        tables = ("facts", "private_memories", "episodes", "corrections", "exchanges")
        deleted: dict[str, int] = {}
        target = str(scope_id or "").strip()
        with self._connection() as connection:
            for table in tables:
                cursor = connection.execute(
                    f"DELETE FROM {table} WHERE scope_id = ?",
                    (target,),
                )
                deleted[table] = int(cursor.rowcount or 0)
        return deleted
