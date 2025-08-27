#!/usr/bin/env python3
"""
MongoDB Connection Diagnostic Script

This script helps diagnose MongoDB Atlas connection issues.
Run this to test your MongoDB connection independently of the bot.
"""

import os
import sys
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    """Test MongoDB connection with detailed diagnostics"""
    print("🔍 MongoDB Connection Diagnostic Tool")
    print("=" * 50)

    # Get MongoDB URI
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("❌ ERROR: MONGODB_URI not found in environment variables")
        print("   Please check your .env file")
        return False

    print(f"📍 MongoDB URI: {mongodb_uri[:50]}...")

    # Connection parameters
    connect_timeout = 30000  # 30 seconds
    server_selection_timeout = 30000  # 30 seconds
    socket_timeout = 20000  # 20 seconds

    print("⏱️  Connection timeouts:")
    print(f"   - Connect timeout: {connect_timeout}ms")
    print(f"   - Server selection timeout: {server_selection_timeout}ms")
    print(f"   - Socket timeout: {socket_timeout}ms")

    try:
        print("\n🔌 Attempting to connect...")

        # Create client with explicit timeouts
        client = MongoClient(
            mongodb_uri,
            connectTimeoutMS=connect_timeout,
            serverSelectionTimeoutMS=server_selection_timeout,
            socketTimeoutMS=socket_timeout,
            maxPoolSize=5,
            minPoolSize=1,
            retryWrites=True,
            retryReads=True
        )

        # Test connection with ping
        print("🏓 Testing connection with ping...")
        start_time = time.time()
        client.admin.command("ping")
        ping_time = time.time() - start_time

        print(f"🏓 Ping successful in {ping_time:.2f}s")
        print(f"📊 Server info: {client.server_info()['version']}")

        # Get database list
        db_names = client.list_database_names()
        print(f"📂 Available databases: {db_names}")

        # Test specific database
        db_name = os.getenv("DB_NAME", "telegram_gemini_bot")
        db = client[db_name]
        collections = db.list_collection_names()
        print(f"📋 Collections in '{db_name}': {collections}")

        print("\n✅ SUCCESS: MongoDB connection is working!")
        print("   Your bot should be able to connect normally now.")

        client.close()
        return True

    except ServerSelectionTimeoutError as e:
        print("\n❌ CONNECTION TIMEOUT ERROR")
        print("   This usually means:")
        print("   1. MongoDB Atlas cluster is PAUSED (common with free tier)")
        print("   2. Network connectivity issues")
        print("   3. Firewall blocking connections")
        print("   4. IP address not whitelisted in MongoDB Atlas")
        print("\n🔧 SOLUTIONS:")
        print("   1. Check MongoDB Atlas dashboard - cluster might be paused")
        print("   2. Add your IP address to the IP whitelist in Atlas")
        print("   3. Try connecting from a different network")
        print("   4. Check if your firewall is blocking outbound connections")
        print(f"\n   Error details: {str(e)}")

    except ConnectionFailure as e:
        print("\n❌ CONNECTION FAILURE")
        print("   This usually means:")
        print("   1. Invalid connection string")
        print("   2. Authentication failure")
        print("   3. Network routing issues")
        print("\n🔧 SOLUTIONS:")
        print("   1. Verify your MONGODB_URI in .env file")
        print("   2. Check username/password in connection string")
        print("   3. Ensure the database user has proper permissions")
        print(f"\n   Error details: {str(e)}")

    except Exception as e:
        print("\n❌ UNEXPECTED ERROR")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check your internet connection")
        print("   2. Verify MongoDB Atlas cluster is running")
        print("   3. Check MongoDB Atlas network access settings")

    return False

def main():
    """Main function"""
    print("Starting MongoDB connection diagnostics...\n")

    success = test_mongodb_connection()

    if not success:
        print("\n" + "=" * 50)
        print("💡 QUICK FIXES:")
        print("1. If using MongoDB Atlas free tier, check if cluster is paused")
        print("2. Add 'DEV_MODE=true' and 'IGNORE_DB_ERROR=true' to your .env file")
        print("3. This will use a mock database for development")
        print("4. Remove these when MongoDB connection is restored")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
