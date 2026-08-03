from __future__ import annotations

import unittest

from astrbot_plugin_botmesh_memory.core import (
    build_identity_system_block,
    sanitize_contexts,
)


class IdentityTests(unittest.TestCase):
    def test_builds_identity_block_from_botmesh_payload(self):
        identity = {
            "account_label": "Rev 账号",
            "self_identity": "蔚来",
            "soul_identity": "蔚来",
            "body_identity": "莉芙",
            "locked": True,
        }
        block = build_identity_system_block(identity, scope_id="soul-swap")

        self.assertEqual(identity["self_identity"], "蔚来")
        self.assertIn("当前灵魂/操控者：蔚来", block)
        self.assertIn("当前身体身份：莉芙", block)
        self.assertIn("防历史覆盖：开启", block)
        self.assertIn("BotMesh Persona", block)


class ContextSanitizerTests(unittest.TestCase):
    def test_removes_thinking_quotes_and_reminders_then_caps_history(self):
        contexts = [
            {"role": "user", "content": f"消息{i}"} for i in range(6)
        ]
        contexts.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "text": "不可回注的推理", "encrypted": "x"},
                    {
                        "type": "text",
                        "text": (
                            "正文<Quoted Message>旧引用</Quoted Message>"
                            "<system_reminder>旧提醒</system_reminder>"
                        ),
                        "encrypted": "y",
                    },
                ],
            }
        )

        cleaned, stats = sanitize_contexts(
            contexts,
            max_items=4,
            remove_thinking=True,
            remove_quotes=True,
            remove_reminders=True,
        )

        self.assertEqual(len(cleaned), 4)
        self.assertEqual(stats["removed_think_parts"], 1)
        self.assertEqual(stats["trimmed_items"], 3)
        last_part = cleaned[-1]["content"][0]
        self.assertNotIn("旧引用", last_part["text"])
        self.assertNotIn("旧提醒", last_part["text"])
        self.assertNotIn("encrypted", last_part)


if __name__ == "__main__":
    unittest.main()
