"""
Migration: Add index on cache_key field for faster lookups

This improves performance of delete operations that search by cache_key pattern.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_database
import logging

logger = logging.getLogger(__name__)


def add_cache_key_index():
    """Add index on cache_key field in conversations collection"""
    db, client = get_database()

    if db is None:
        logger.error("Failed to connect to database")
        return False

    try:
        conversations = db.conversations

        existing_indexes = list(conversations.list_indexes())
        index_names = [idx["name"] for idx in existing_indexes]
        logger.info(f"Existing indexes: {index_names}")

        # Check if cache_key index already exists
        if "cache_key_1" in index_names:
            logger.info("✅ cache_key index already exists - no action needed")
            logger.info("Database is already optimized for fast cache_key lookups")
            return True

        # Create index if it doesn't exist
        result = conversations.create_index("cache_key", unique=False)
        logger.info(f"✅ Index created: {result}")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating index: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    add_cache_key_index()
