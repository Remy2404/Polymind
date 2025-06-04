import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Import the classes we're testing
from src.services.group_chat.group_manager import (
    GroupManager,
    GroupConversationContext,
    GroupParticipant,
    GroupRole,
)
from src.services.group_chat.ui_components import GroupUIManager
from src.services.group_chat.integration import GroupChatIntegration


class TestGroupManager(unittest.TestCase):
    """Test cases for GroupManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user_data_manager = Mock()
        self.mock_conversation_manager = Mock()
        self.group_manager = GroupManager(
            self.mock_user_data_manager, self.mock_conversation_manager
        )

    def test_group_manager_initialization(self):
        """Test that GroupManager initializes correctly."""
        self.assertIsInstance(self.group_manager, GroupManager)
        self.assertEqual(len(self.group_manager.group_contexts), 0)
        self.assertEqual(len(self.group_manager.active_threads), 0)

    async def test_group_context_creation(self):
        """Test creation of new group context."""
        # Mock chat and user objects
        mock_chat = Mock()
        mock_chat.id = -123456789
        mock_chat.title = "Test Group"
        mock_chat.type = "group"

        mock_user = Mock()
        mock_user.id = 123456
        mock_user.username = "testuser"
        mock_user.full_name = "Test User"
        mock_user.first_name = "Test"

        # Mock user data manager response
        self.mock_user_data_manager.get_user_data = AsyncMock(return_value={})

        # Create group context
        group_context = await self.group_manager._get_or_create_group_context(
            mock_chat, mock_user
        )

        # Verify group context creation
        self.assertIsInstance(group_context, GroupConversationContext)
        self.assertEqual(group_context.group_id, -123456789)
        self.assertEqual(group_context.group_name, "Test Group")
        self.assertIn(123456, group_context.participants)

        # Verify participant creation
        participant = group_context.participants[123456]
        self.assertIsInstance(participant, GroupParticipant)
        self.assertEqual(participant.user_id, 123456)
        self.assertEqual(participant.username, "testuser")
        self.assertEqual(participant.role, GroupRole.MEMBER)

    async def test_group_message_processing(self):
        """Test processing of group messages."""
        # Create mock objects
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user = Mock()
        mock_message = Mock()

        # Set up mock data
        mock_chat.id = -123456789
        mock_chat.title = "Test Group"
        mock_chat.type = "group"

        mock_user.id = 123456
        mock_user.username = "testuser"
        mock_user.full_name = "Test User"

        mock_message.from_user = mock_user
        mock_message.text = "Hello, this is a test message"
        mock_message.reply_to_message = None

        mock_update.effective_chat = mock_chat
        mock_update.effective_user = mock_user
        mock_update.message = mock_message

        # Mock dependencies
        self.mock_user_data_manager.get_user_data = AsyncMock(return_value={})

        # Process group message
        enhanced_message, metadata = await self.group_manager.handle_group_message(
            mock_update, mock_context, "Hello, this is a test message"
        )

        # Verify response
        self.assertIsInstance(enhanced_message, str)
        self.assertIsInstance(metadata, dict)
        self.assertIn("group_id", metadata)
        self.assertEqual(metadata["group_id"], -123456789)

    async def test_thread_creation(self):
        """Test conversation thread creation."""
        # Set up group context
        group_context = GroupConversationContext(
            group_id=-123456789,
            group_name="Test Group",
            shared_memory={},
            active_topics=[],
            recent_messages=[],
            participants={},
            threads={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Create mock message with reply
        mock_message = Mock()
        mock_message.from_user = Mock()
        mock_message.from_user.id = 123456
        mock_message.reply_to_message = Mock()
        mock_message.reply_to_message.message_id = 789

        # Get conversation thread
        thread = await self.group_manager._get_conversation_thread(
            group_context, mock_message
        )

        # Verify thread creation
        self.assertIsNotNone(thread)
        self.assertEqual(thread.group_id, -123456789)
        self.assertIn(123456, thread.participants)

    async def test_group_analytics(self):
        """Test group analytics generation."""
        # Set up group context with data
        group_context = GroupConversationContext(
            group_id=-123456789,
            group_name="Test Group",
            shared_memory={"key1": "value1", "key2": "value2"},
            active_topics=["python", "coding", "bot"],
            recent_messages=[],
            participants={
                123456: GroupParticipant(
                    user_id=123456,
                    username="user1",
                    full_name="User One",
                    role=GroupRole.MEMBER,
                    join_date=datetime.now(),
                    last_active=datetime.now(),
                    message_count=10,
                ),
                789012: GroupParticipant(
                    user_id=789012,
                    username="user2",
                    full_name="User Two",
                    role=GroupRole.ADMIN,
                    join_date=datetime.now(),
                    last_active=datetime.now(),
                    message_count=25,
                ),
            },
            threads={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Add to group contexts
        self.group_manager.group_contexts[-123456789] = group_context

        # Get analytics
        analytics = await self.group_manager.get_group_analytics(-123456789)

        # Verify analytics
        self.assertIsInstance(analytics, dict)
        self.assertEqual(analytics["group_name"], "Test Group")
        self.assertEqual(analytics["total_participants"], 2)
        self.assertEqual(analytics["total_messages"], 35)
        self.assertEqual(analytics["shared_memory_items"], 2)
        self.assertEqual(len(analytics["active_topics"]), 3)


class TestGroupUIManager(unittest.TestCase):
    """Test cases for GroupUIManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui_manager = GroupUIManager()

    def test_ui_manager_initialization(self):
        """Test that GroupUIManager initializes correctly."""
        self.assertIsInstance(self.ui_manager, GroupUIManager)
        self.assertIn("group", self.ui_manager.EMOJIS)
        self.assertIn("thread", self.ui_manager.EMOJIS)

    async def test_analytics_formatting(self):
        """Test formatting of group analytics."""
        analytics_data = {
            "group_name": "Test Group",
            "total_participants": 5,
            "active_participants": 3,
            "total_messages": 150,
            "active_topics": ["python", "bot", "ai"],
            "active_threads": 2,
            "shared_memory_items": 10,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
        }

        formatted_analytics = await self.ui_manager.format_group_analytics(
            analytics_data
        )

        # Verify formatting
        self.assertIsInstance(formatted_analytics, str)
        self.assertIn("Test Group", formatted_analytics)
        self.assertIn("5", formatted_analytics)  # total participants
        self.assertIn("150", formatted_analytics)  # total messages
        self.assertIn("python", formatted_analytics)  # topics

    async def test_thread_list_formatting(self):
        """Test formatting of thread list."""
        threads_data = {
            "thread1": {
                "topic": "Python Discussion",
                "participants": [123, 456, 789],
                "message_count": 25,
                "last_message_at": datetime.now().isoformat(),
                "is_active": True,
            },
            "thread2": {
                "topic": "Bot Development",
                "participants": [123, 456],
                "message_count": 15,
                "last_message_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "is_active": True,
            },
        }

        formatted_threads = await self.ui_manager.format_thread_list(threads_data)

        # Verify formatting
        self.assertIsInstance(formatted_threads, str)
        self.assertIn("Python Discussion", formatted_threads)
        self.assertIn("Bot Development", formatted_threads)
        self.assertIn("25 messages", formatted_threads)

    async def test_settings_menu_creation(self):
        """Test creation of settings menu."""
        settings_menu = await self.ui_manager.create_settings_menu(-123456789)

        # Verify menu creation
        self.assertIsInstance(settings_menu, str)
        self.assertIn("Group Chat Settings", settings_menu)
        self.assertIn("/groupstats", settings_menu)
        self.assertIn("Shared Memory", settings_menu)


class TestGroupChatIntegration(unittest.TestCase):
    """Test cases for GroupChatIntegration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_user_data_manager = Mock()
        self.mock_conversation_manager = Mock()
        self.integration = GroupChatIntegration(
            self.mock_user_data_manager, self.mock_conversation_manager
        )

    def test_integration_initialization(self):
        """Test that GroupChatIntegration initializes correctly."""
        self.assertIsInstance(self.integration, GroupChatIntegration)
        self.assertIsNotNone(self.integration.group_manager)
        self.assertIsNotNone(self.integration.ui_manager)

    async def test_group_message_processing(self):
        """Test integration's group message processing."""
        # Create mock objects
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user = Mock()

        # Set up mock data
        mock_chat.id = -123456789
        mock_chat.title = "Test Group"
        mock_chat.type = "group"

        mock_user.id = 123456
        mock_user.username = "testuser"
        mock_user.full_name = "Test User"

        mock_update.effective_chat = mock_chat
        mock_update.effective_user = mock_user

        # Mock dependencies
        self.mock_user_data_manager.get_user_data = AsyncMock(return_value={})

        # Process group message
        enhanced_message, metadata = await self.integration.process_group_message(
            mock_update, mock_context, "Test message"
        )

        # Verify response
        self.assertIsInstance(enhanced_message, str)
        self.assertIsInstance(metadata, dict)
        self.assertTrue(metadata.get("group_features_enabled", False))

    async def test_group_command_handling(self):
        """Test handling of group commands."""
        # Create mock objects
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user = Mock()

        # Set up mock data
        mock_chat.id = -123456789
        mock_chat.type = "group"

        mock_user.id = 123456

        mock_update.effective_chat = mock_chat
        mock_update.effective_user = mock_user

        # Test groupstats command
        response = await self.integration.handle_group_command(
            mock_update, mock_context, "groupstats", []
        )

        # Verify response (should be a formatted analytics message)
        self.assertIsInstance(response, str)

    def test_group_chat_detection(self):
        """Test detection of group chats."""
        # Test group chat
        mock_chat = Mock()
        mock_chat.type = "group"
        self.assertTrue(self.integration.is_group_chat(mock_chat))

        # Test supergroup
        mock_chat.type = "supergroup"
        self.assertTrue(self.integration.is_group_chat(mock_chat))

        # Test private chat
        mock_chat.type = "private"
        self.assertFalse(self.integration.is_group_chat(mock_chat))


class TestGroupChatFeatures(unittest.TestCase):
    """Integration tests for complete group chat features."""

    def setUp(self):
        """Set up complete test environment."""
        self.mock_user_data_manager = Mock()
        self.mock_conversation_manager = Mock()

        # Mock database responses
        self.mock_user_data_manager.get_user_data = AsyncMock(return_value={})
        self.mock_user_data_manager.update_user_data = AsyncMock(return_value=True)

        # Create integration
        self.integration = GroupChatIntegration(
            self.mock_user_data_manager, self.mock_conversation_manager
        )

    async def test_complete_group_conversation_flow(self):
        """Test a complete group conversation flow."""
        # Create mock group conversation
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user1 = Mock()
        mock_user2 = Mock()

        # Set up group data
        mock_chat.id = -123456789
        mock_chat.title = "Test Development Group"
        mock_chat.type = "group"

        mock_user1.id = 123456
        mock_user1.username = "developer1"
        mock_user1.full_name = "Developer One"

        mock_user2.id = 789012
        mock_user2.username = "developer2"
        mock_user2.full_name = "Developer Two"

        # Simulate conversation flow
        conversations = [
            (mock_user1, "Hey everyone, let's discuss the new bot features"),
            (mock_user2, "Great! I think we should focus on group chat integration"),
            (mock_user1, "Agreed! The shared memory feature looks promising"),
            (
                mock_user2,
                "Yes, and the thread management will help organize discussions",
            ),
        ]

        for user, message in conversations:
            mock_update.effective_chat = mock_chat
            mock_update.effective_user = user

            # Process each message
            enhanced_message, metadata = await self.integration.process_group_message(
                mock_update, mock_context, message
            )

            # Verify each response
            self.assertIsInstance(enhanced_message, str)
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata["group_id"], -123456789)

        # Verify group context was created and updated
        group_manager = self.integration.group_manager
        self.assertIn(-123456789, group_manager.group_contexts)

        group_context = group_manager.group_contexts[-123456789]
        self.assertEqual(len(group_context.participants), 2)
        self.assertGreater(len(group_context.recent_messages), 0)
        self.assertGreater(len(group_context.active_topics), 0)

    async def test_group_analytics_and_ui(self):
        """Test group analytics generation and UI formatting."""
        # Set up group with some data
        group_id = -123456789

        # Add some mock data to the group
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user = Mock()

        mock_chat.id = group_id
        mock_chat.title = "Analytics Test Group"
        mock_chat.type = "group"

        mock_user.id = 123456
        mock_user.username = "testuser"
        mock_user.full_name = "Test User"

        mock_update.effective_chat = mock_chat
        mock_update.effective_user = mock_user

        # Process a few messages to generate data
        messages = [
            "Let's test the analytics feature",
            "This should create some topics and activity",
            "Analytics data generation is important for insights",
        ]

        for message in messages:
            await self.integration.process_group_message(
                mock_update, mock_context, message
            )

        # Get analytics
        analytics = await self.integration.group_manager.get_group_analytics(group_id)

        # Format analytics with UI manager
        formatted_analytics = await self.integration.ui_manager.format_group_analytics(
            analytics
        )

        # Verify analytics and formatting
        self.assertIsInstance(analytics, dict)
        self.assertIsInstance(formatted_analytics, str)
        self.assertIn("Analytics Test Group", formatted_analytics)
        self.assertIn("Total:", formatted_analytics)

    async def test_thread_management(self):
        """Test conversation thread management."""
        # Set up group conversation with replies
        mock_update = Mock()
        mock_context = Mock()
        mock_chat = Mock()
        mock_user1 = Mock()
        mock_user2 = Mock()
        mock_message = Mock()
        mock_reply_message = Mock()

        # Set up basic data
        mock_chat.id = -123456789
        mock_chat.title = "Thread Test Group"
        mock_chat.type = "group"

        mock_user1.id = 123456
        mock_user1.username = "user1"
        mock_user1.full_name = "User One"

        mock_user2.id = 789012
        mock_user2.username = "user2"
        mock_user2.full_name = "User Two"

        # First message (creates initial context)
        mock_message.from_user = mock_user1
        mock_message.text = "Let's start a discussion about threads"
        mock_message.reply_to_message = None

        mock_update.effective_chat = mock_chat
        mock_update.effective_user = mock_user1
        mock_update.message = mock_message

        await self.integration.process_group_message(
            mock_update, mock_context, "Let's start a discussion about threads"
        )

        # Reply message (creates thread)
        mock_reply_message.from_user = mock_user2
        mock_reply_message.text = "Great idea! Thread management is crucial"
        mock_reply_message.reply_to_message = Mock()
        mock_reply_message.reply_to_message.message_id = 1001

        mock_update.effective_user = mock_user2
        mock_update.message = mock_reply_message

        await self.integration.process_group_message(
            mock_update, mock_context, "Great idea! Thread management is crucial"
        )

        # Verify thread creation
        group_manager = self.integration.group_manager
        group_context = group_manager.group_contexts[-123456789]

        self.assertGreater(len(group_context.threads), 0)

        # Test thread listing
        formatted_threads = await self.integration.ui_manager.format_thread_list(
            group_context.threads
        )

        self.assertIsInstance(formatted_threads, str)
        self.assertIn("conversation", formatted_threads.lower())


async def run_tests():
    """Run all tests asynchronously."""
    import sys

    # Create test suite
    test_classes = [
        TestGroupManager,
        TestGroupUIManager,
        TestGroupChatIntegration,
        TestGroupChatFeatures,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    print("🧪 Running Group Chat Integration Tests...")
    print("=" * 50)

    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}...")

        # Get all test methods
        test_methods = [
            method for method in dir(test_class) if method.startswith("test_")
        ]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setUp()

                # Get the test method
                method = getattr(test_instance, test_method)

                # Run async or sync test
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()

                print(f"  ✅ {test_method}")
                passed_tests += 1

            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
                failed_tests.append((test_class.__name__, test_method, str(e)))

    # Print summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for class_name, method_name, error in failed_tests:
            print(f"   {class_name}.{method_name}: {error}")
    else:
        print(f"\n🎉 All tests passed!")

    return len(failed_tests) == 0


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_tests())

    if not success:
        exit(1)

    print("\n✅ Group Chat Integration tests completed successfully!")
    print("🚀 The group chat features are ready for deployment!")
