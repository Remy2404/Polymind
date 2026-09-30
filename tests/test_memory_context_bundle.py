import unittest
from unittest.mock import patch

from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager


class TestMemoryContextBundle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch(
            "src.services.memory_context.memory_manager.get_database",
            return_value=(None, None),
        ):
            self.memory_manager = MemoryManager(db=None)
        self.model_history_manager = ModelHistoryManager(self.memory_manager)
        self.user_id = 101
        self.model_id = "gemini"

    async def test_context_bundle_provides_summary_and_highlights(self):
        for i in range(24):
            await self.model_history_manager.save_message_pair(
                self.user_id,
                f"Remember the project milestone {i}",
                f"Recorded milestone {i}",
                model_id=self.model_id,
            )

        conversation_id = self.model_history_manager._get_conversation_id(
            self.user_id, self.model_id
        )
        bundle = await self.memory_manager.build_context_bundle(
            conversation_id, limit=12
        )

        self.assertGreaterEqual(len(bundle["recent"]), 12)
        self.assertIsNotNone(bundle["summary"])
        self.assertTrue(
            any(
                msg.get("metadata", {}).get("context_type") == "highlight"
                for msg in bundle["highlights"]
            )
        )

        history = await self.model_history_manager.get_history(
            self.user_id, max_messages=12, model_id=self.model_id
        )
        summary_entries = [
            msg
            for msg in history
            if msg.get("metadata", {}).get("context_type") == "summary"
        ]
        highlight_entries = [
            msg
            for msg in history
            if msg.get("metadata", {}).get("context_type") == "highlight"
        ]
        self.assertTrue(summary_entries)
        self.assertTrue(highlight_entries)

    async def test_history_respects_limit(self):
        for i in range(30):
            await self.model_history_manager.save_message_pair(
                self.user_id,
                f"Task {i}: finalize integration and remember testing",
                f"Logged task {i} completion",
                model_id=self.model_id,
            )

        history = await self.model_history_manager.get_history(
            self.user_id, max_messages=8, model_id=self.model_id
        )
        non_summary_messages = [
            msg
            for msg in history
            if msg.get("metadata", {}).get("context_type") != "summary"
        ]
        self.assertLessEqual(len(non_summary_messages), 8)
        summary_count = sum(
            1
            for msg in history
            if msg.get("metadata", {}).get("context_type") == "summary"
        )
        self.assertLessEqual(summary_count, 1)


if __name__ == "__main__":
    unittest.main()
