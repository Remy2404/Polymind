import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import time


def test_database_connection():
    # Load environment variables with explicit encoding
    load_dotenv(encoding="utf-8")

    uri = os.getenv("DATABASE_URL")

    print("🔄 Connecting to database...")

    max_retries = 3
    current_retry = 0

    while current_retry < max_retries:
        try:
            # Create a MongoDB client with Server API version 1
            client = MongoClient(uri, server_api=ServerApi("1"))

            # Test connection with ping
            client.admin.command("ping")
            print("✅ Successfully connected to MongoDB!")

            # Get database reference
            db = client["gembot"]
            print("📚 Available collections:", db.list_collection_names())

            client.close()
            return True

        except Exception as e:
            print(f"❌ Connection attempt {current_retry + 1} failed: {str(e)}")
            current_retry += 1
            if current_retry < max_retries:
                print("⏳ Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("❌ Max retries reached. Could not connect to database.")
                return False


if __name__ == "__main__":
    test_database_connection()
