import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get MongoDB connection string from environment variable
MONGODB_URI = os.getenv('DATABASE_URL')

def get_database():
    if not MONGODB_URI:
        logger.error("DATABASE_URL environment variable is not set")
        return None, None

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")

        db_name = os.getenv('MONGODB_DB_NAME', 'gembot')
        db = client[db_name]
        return db, client
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to MongoDB: {e}")
        return None, None

def close_database_connection(client):
    """
    Closes the MongoDB client connection.

    Args:
        client (MongoClient): The MongoDB client instance to close.
    """
    if client:
        try:
            client.close()
            logger.info("MongoDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

def get_image_cache_collection(db):
    if db is not None:
        return db.image_cache
    else:
        logger.error("Database connection is not established.")
        return None