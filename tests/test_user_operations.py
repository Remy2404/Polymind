import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def test_user_operations():
    # Load environment variables with explicit encoding
    load_dotenv(encoding="utf-8")

    # Get database connection string
    uri = os.getenv("DATABASE_URL")

    # Initialize MongoDB client
    client = MongoClient(uri, server_api=ServerApi("1"))
    db = client["gembot"]
    users_collection = db["users"]

    # Test user operations
    test_user = {
        "user_id": "test123",
        "username": "testuser",
        "first_name": "Test",
        "last_name": "User",
    }

    # Insert test user
    result = users_collection.insert_one(test_user)
    print(f"✅ Inserted test user with id: {result.inserted_id}")

    # Find the user
    found_user = users_collection.find_one({"user_id": "test123"})
    print(f"📄 Found user: {found_user}")

    # Clean up - delete test user
    users_collection.delete_one({"user_id": "test123"})
    print("🗑️ Cleaned up test user")

    client.close()


if __name__ == "__main__":
    test_user_operations()
