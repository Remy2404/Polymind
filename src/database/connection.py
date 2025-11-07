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
load_dotenv()
logger = logging.getLogger(__name__)
CONNECTION_POOL = {}
# Production-optimized connection settings
MAX_POOL_SIZE = int(os.getenv("MONGODB_MAX_POOL_SIZE", "50"))  
MIN_POOL_SIZE = int(os.getenv("MONGODB_MIN_POOL_SIZE", "10")) 
CONNECTION_TIMEOUT = int(os.getenv("MONGODB_CONNECTION_TIMEOUT", "30000"))
SERVER_SELECTION_TIMEOUT = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT", "30000"))
SOCKET_TIMEOUT = int(os.getenv("MONGODB_SOCKET_TIMEOUT", "45000"))
MAX_IDLE_TIME_MS = int(os.getenv("MONGODB_MAX_IDLE_TIME", "300000"))
RETRY_WRITES = os.getenv("MONGODB_RETRY_WRITES", "true").lower() == "true"
HEARTBEAT_FREQUENCY = int(os.getenv("MONGODB_HEARTBEAT_FREQUENCY", "30000"))
CONNECT_TIMEOUT = int(os.getenv("MONGODB_CONNECT_TIMEOUT", "30000"))

# Production performance settings
MAX_STALENESS_SECONDS = int(os.getenv("MONGODB_MAX_STALENESS", "90"))
READ_PREFERENCE = os.getenv("MONGODB_READ_PREFERENCE", "primaryPreferred")
WRITE_CONCERN = os.getenv("MONGODB_WRITE_CONCERN", "majority")
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
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    if not mongodb_uri:
        logger.warning("MONGODB_URI not found in environment variables")
        if dev_mode:
            logger.info("Development mode detected, using mock database")
            mock_db = MockDatabase()
            return mock_db, None
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
    current_retry_interval = retry_interval
    for attempt in range(max_retries):
        try:
            import importlib
            if importlib.util.find_spec("snappy") is not None:
                compressors = ["snappy", "zlib"]
            else:
                compressors = ["zlib"]
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
                retryReads=True,
                compressors=compressors,
                zlibCompressionLevel=6,
                # Production performance settings
                maxStalenessSeconds=MAX_STALENESS_SECONDS,
                readPreference=READ_PREFERENCE,
                writeConcern=WRITE_CONCERN,
                # Connection optimization
                waitQueueTimeoutMS=30000,
                maxConnecting=5,
            )
            try:
                client.admin.command("ping", maxTimeMS=15000)
                logger.info("MongoDB connection verified successfully")
            except Exception as ping_error:
                logger.warning(f"Ping verification failed: {ping_error}")
            db = client[db_name]
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            _ensure_indexes(db)
            return db, client
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            error_msg = str(e).lower()
            is_timeout = any(
                keyword in error_msg
                for keyword in ["timeout", "timed out", "connection timeout"]
            )
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
                current_retry_interval = min(current_retry_interval * 2, 30)
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
                if os.getenv("IGNORE_DB_ERROR", "false").lower() == "true":
                    logger.warning("IGNORE_DB_ERROR is true, using mock database")
                    mock_db = MockDatabase()
                    return mock_db, None
                return None, None
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
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
        if "users" in db.list_collection_names():
            db.users.create_index("user_id", unique=True, background=True)
            db.users.create_index("username", background=True)
        if "conversation_history" in db.list_collection_names():
            db.conversation_history.create_index(
                [("user_id", 1), ("timestamp", -1)], background=True
            )
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
