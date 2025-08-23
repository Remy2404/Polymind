import os
import time
import asyncio
import logging
from typing import Tuple, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.collection import Collection
from dotenv import load_dotenv

# Make sure to load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Connection pool configuration with timeout improvements
CONNECTION_POOL = {}
MAX_POOL_SIZE = 20  # Reduced pool size for better stability
MIN_POOL_SIZE = 2   # Reduced minimum connections
CONNECTION_TIMEOUT = 30000  # Increased to 30 seconds
SERVER_SELECTION_TIMEOUT = 30000  # Added server selection timeout
SOCKET_TIMEOUT = 20000  # Added socket timeout for individual operations
MAX_IDLE_TIME_MS = 300000  # Increased to 5 minutes
RETRY_WRITES = True
HEARTBEAT_FREQUENCY = 30000  # Added heartbeat frequency (30 seconds)
CONNECT_TIMEOUT = 30000  # Added connection timeout


# Mock database for development mode
class MockDatabase:
    """Mock database implementation for development without MongoDB"""

    def __init__(self):
        self.collections = {}
        logger.warning("Using mock database (development mode)")

    def __getattr__(self, name):
        """Create mock collections on demand"""
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]

    def list_collection_names(self):
        """Return mock collection names"""
        return list(self.collections.keys())

    def get_collection(self, name):
        """Return a mock collection by name"""
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]


class MockCollection:
    """Mock collection for development mode"""

    def __init__(self, name):
        self.name = name
        self.data = []
        self.indexes = []

    def create_index(self, *args, **kwargs):
        """Mock create_index"""
        self.indexes.append(args)
        return None

    def find_one(self, query=None, *args, **kwargs):
        """Mock find_one"""
        return None

    def insert_one(self, document, *args, **kwargs):
        """Mock insert_one"""
        self.data.append(document)
        return type("", (), {"inserted_id": len(self.data)})

    def find(self, *args, **kwargs):
        """Mock find"""
        return []

    def update_one(self, query, update, *args, **kwargs):
        """Mock update_one to prevent AttributeError"""
        logger.debug(f"Mock update_one called with query: {query}, update: {update}")
        # Return a mock result with acknowledged and matched_count
        return type(
            "UpdateResult",
            (),
            {"acknowledged": True, "matched_count": 1, "modified_count": 1},
        )

    def delete_many(self, query, *args, **kwargs):
        """Mock delete_many"""
        logger.debug(f"Mock delete_many called with query: {query}")
        return type("DeleteResult", (), {"acknowledged": True, "deleted_count": 0})


def get_database(
    max_retries: int = 3, retry_interval: float = 1.0
) -> Tuple[Optional[Database], Optional[MongoClient]]:
    """
    Get a MongoDB database connection with retry logic and connection pooling

    Args:
        max_retries: Maximum number of connection retry attempts
        retry_interval: Time between retry attempts in seconds

    Returns:
        Tuple containing (database, client) or (None, None) if connection fails
    """
    mongodb_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME", "telegram_gemini_bot")

    # Check if we're in development mode
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    if not mongodb_uri:
        logger.warning("MONGODB_URI not found in environment variables")

        # Check for development mode or fallback to mock DB
        if dev_mode:
            logger.info("Development mode detected, using mock database")
            mock_db = MockDatabase()
            return mock_db, None

        # Try alternate environment variable names
        mongodb_uri = os.getenv("MONGO_URI") or os.getenv("MONGO_URL")
        if not mongodb_uri:
            logger.error(
                "No MongoDB connection string found in any environment variable"
            )
            if os.getenv("IGNORE_DB_ERROR", "false").lower() == "true":
                logger.warning("IGNORE_DB_ERROR is true, using mock database")
                mock_db = MockDatabase()
                return mock_db, None
            else:
                return None, None

    # Use exponential backoff for retries
    current_retry_interval = retry_interval

    for attempt in range(max_retries):
        try:
            # Create client with optimized connection pool settings for timeout handling
            # Note: Compression libraries are optional and won't cause failures
            try:
                import snappy  # noqa: F401 - Only checking availability
                compressors = ['snappy', 'zlib']
            except ImportError:
                compressors = ['zlib']  # Fallback to zlib only if snappy not available
                
            client = MongoClient(
                mongodb_uri,
                maxPoolSize=MAX_POOL_SIZE,
                minPoolSize=MIN_POOL_SIZE,
                connectTimeoutMS=CONNECT_TIMEOUT,
                serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT,
                socketTimeoutMS=SOCKET_TIMEOUT,
                maxIdleTimeMS=MAX_IDLE_TIME_MS,
                heartbeatFrequencyMS=HEARTBEAT_FREQUENCY,
                retryWrites=RETRY_WRITES,
                retryReads=True,  # Enable retry reads for better resilience
                compressors=compressors,  # Use available compressors
                zlibCompressionLevel=6,  # Moderate compression level
            )

            # Force a connection to verify it works with proper timeout handling
            try:
                # Use ping instead of ismaster (deprecated) with explicit timeout
                client.admin.command("ping", maxTimeMS=15000)  # 15 second ping timeout
                logger.info("MongoDB connection verified successfully")
            except Exception as ping_error:
                logger.warning(f"Ping verification failed: {ping_error}")
                # Continue anyway as connection might still work

            # Get the database
            db = client[db_name]
            logger.info(f"Successfully connected to MongoDB database: {db_name}")

            # Create indexes for common queries if they don't exist
            _ensure_indexes(db)

            return db, client

        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            # Enhanced error handling with specific timeout error detection
            error_msg = str(e).lower()
            is_timeout = any(keyword in error_msg for keyword in ['timeout', 'timed out', 'connection timeout'])
            
            if attempt < max_retries - 1:
                if is_timeout:
                    logger.warning(
                        f"Database connection timeout on attempt {attempt + 1}: {str(e)[:200]}... "
                        f"This is likely due to network issues or MongoDB Atlas connection limits. "
                        f"Retrying in {current_retry_interval:.1f}s..."
                    )
                else:
                    logger.warning(
                        f"Database connection attempt {attempt + 1} failed: {str(e)[:200]}... "
                        f"Retrying in {current_retry_interval:.1f}s..."
                    )
                time.sleep(current_retry_interval)
                current_retry_interval = min(current_retry_interval * 2, 30)  # Cap at 30 seconds
            else:
                if is_timeout:
                    logger.error(
                        f"Failed to connect to MongoDB after {max_retries} attempts due to persistent timeouts. "
                        f"This may be due to: 1) Network connectivity issues, 2) MongoDB Atlas connection limits, "
                        f"3) Firewall blocking connections, or 4) MongoDB server overload. "
                        f"Last error: {str(e)}"
                    )
                else:
                    logger.error(
                        f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}"
                    )
                # Use mock DB as fallback if configured
                if os.getenv("IGNORE_DB_ERROR", "false").lower() == "true":
                    logger.warning("IGNORE_DB_ERROR is true, using mock database")
                    mock_db = MockDatabase()
                    return mock_db, None
                return None, None
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            # Use mock DB as fallback if configured
            if os.getenv("IGNORE_DB_ERROR", "false").lower() == "true":
                logger.warning("IGNORE_DB_ERROR is true, using mock database")
                mock_db = MockDatabase()
                return mock_db, None
            return None, None


def _ensure_indexes(db: Database) -> None:
    """
    Create indexes on commonly queried fields to improve performance
    """
    try:
        # User collection indexes
        if "users" in db.list_collection_names():
            db.users.create_index("user_id", unique=True, background=True)
            db.users.create_index("username", background=True)

        # Conversation history indexes
        if "conversation_history" in db.list_collection_names():
            db.conversation_history.create_index(
                [("user_id", 1), ("timestamp", -1)], background=True
            )

        # Document history indexes
        if "document_history" in db.list_collection_names():
            db.document_history.create_index(
                [("user_id", 1), ("timestamp", -1)], background=True
            )

        logger.info("Database indexes created or confirmed")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")


def close_database_connection(client: Optional[MongoClient]) -> None:
    """
    Safely close a MongoDB client connection

    Args:
        client: MongoDB client to close
    """
    if client is not None:
        try:
            client.close()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")


async def get_database_async(
    max_retries: int = 3, retry_interval: float = 1.0
) -> Tuple[Optional[Database], Optional[MongoClient]]:
    """
    Asynchronous version of get_database for use in async code

    This runs the potentially blocking MongoDB connection in a thread pool
    to avoid blocking the event loop
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: get_database(max_retries, retry_interval)
    )


def get_image_cache_collection(db: Optional[Database]) -> Optional[Collection]:
    """Get the image cache collection"""
    if db is None:
        return None
    return db.image_cache
