#!/usr/bin/env python3
"""Test the integration between agent.py and message_handlers.py"""

import sys
import asyncio
sys.path.append('src')

async def test_integration():
    try:
        # Test imports
        from handlers.message_handlers import MessageHandlers
        from services.agent import EnhancedAgent, AgentDeps
        from services.model_handlers.model_configs import get_default_agent_model
        print('✓ All imports successful')
        
        # Test model selection
        default_model = get_default_agent_model()
        print(f'✓ Default model: {default_model.model_id} ({default_model.display_name})')
        
        # Test agent creation
        agent = EnhancedAgent(preferred_model='qwen3-14b')
        print('✓ Enhanced Agent created successfully')
        
        # Test mock message handler
        class MockAPI:
            def __init__(self, name):
                self.name = name
        
        class MockLogger:
            def log_message(self, msg, user_id):
                pass
        
        from services.user_data_manager import UserDataManager
        
        gemini_api = MockAPI('gemini')
        user_data_manager = UserDataManager()
        telegram_logger = MockLogger()
        text_handler = MockAPI('text_handler')
        
        message_handlers = MessageHandlers(
            gemini_api=gemini_api,
            user_data_manager=user_data_manager,
            telegram_logger=telegram_logger,
            text_handler=text_handler
        )
        print('✓ MessageHandlers initialized successfully')
        
        # Test research feature detection
        test_queries = [
            'What is Python?',
            'Search for latest AI news', 
            'How to use FastAPI?',
            'Hello there!',
            'Company information about Tesla'
        ]
        
        print('\n--- Research Feature Detection Test ---')
        for query in test_queries:
            should_research = await message_handlers.can_use_research_features(query)
            status = "🔍 Research" if should_research else "💬 Chat"
            print(f'{status}: "{query}"')
        
        print('\n✅ All integration tests passed!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integration())
