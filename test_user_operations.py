from src.services.user_data_manager import UserDataManager
import os
from dotenv import load_dotenv

def test_user_operations():
    load_dotenv()
    
    try:
        print("🔄 Initializing UserDataManager...")
        user_manager = UserDataManager()
        print("✅ UserDataManager initialized successfully!")
        
        # Test user initialization
        test_user_id = 12345
        print(f"🔄 Testing user initialization for user_id: {test_user_id}")
        user_manager.initialize_user(test_user_id)
        print("✅ User initialized successfully!")
        
        # Test getting user settings
        print("🔄 Testing get user settings...")
        settings = user_manager.get_user_settings(test_user_id)
        print(f"✅ User settings retrieved: {settings}")
        
        # Test updating user stats
        print("🔄 Testing update user stats...")
        user_manager.update_user_stats(test_user_id)
        stats = user_manager.get_user_statistics(test_user_id)
        print(f"✅ User stats updated: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_user_operations() 