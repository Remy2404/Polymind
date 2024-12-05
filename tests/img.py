import pytest
import mongomock
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.services.user_data_manager import UserDataManager

class TestUserDataManager:
    @pytest.fixture
    def user_data_manager(self):
        # Setup mock MongoDB client
        mock_client = mongomock.MongoClient()
        mock_db = mock_client.test_db
        mock_collection = mock_db.users
        
        # Create manager instance with mocked collection
        manager = UserDataManager()
        manager.users_collection = mock_collection
        manager.logger = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_update_stats_text_message(self, user_data_manager):
        test_user_id = "123"
        frozen_time = datetime(2024, 1, 1, 12, 0)
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = frozen_time
            await user_data_manager.update_stats(test_user_id, text_message=True)
            
            user_data = user_data_manager.get_user_data(test_user_id)
            assert user_data['stats']['messages'] == 1
            assert user_data['stats']['last_active'] == frozen_time.isoformat()

    @pytest.mark.asyncio
    async def test_update_stats_generated_image(self, user_data_manager):
        test_user_id = "123"
        await user_data_manager.update_stats(test_user_id, generated_images=True)
        
        user_data = user_data_manager.get_user_data(test_user_id)
        assert user_data['stats']['generated_images'] == 1

    @pytest.mark.asyncio
    async def test_update_stats_multiple_types(self, user_data_manager):
        test_user_id = "123"
        await user_data_manager.update_stats(
            test_user_id,
            text_message=True,
            voice_message=True,
            image=True,
            generated_images=True
        )
        
        stats = user_data_manager.get_user_data(test_user_id)['stats']
        assert stats['messages'] == 1
        assert stats['voice_messages'] == 1
        assert stats['images'] == 1
        assert stats['generated_images'] == 1

    @pytest.mark.asyncio
    async def test_update_stats_error_handling(self, user_data_manager):
        test_user_id = "123"
        user_data_manager.users_collection.update_one = MagicMock(side_effect=Exception("Test error"))
        
        await user_data_manager.update_stats(test_user_id, text_message=True)
        user_data_manager.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_initialization(self, user_data_manager):
        test_user_id = "123"
        await user_data_manager.update_stats(test_user_id)
        
        stats = user_data_manager.get_user_data(test_user_id)['stats']
        assert stats['messages'] == 0
        assert stats['voice_messages'] == 0
        assert stats['images'] == 0
        assert stats['generated_images'] == 0