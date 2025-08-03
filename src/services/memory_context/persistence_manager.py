"""
Persistence Management Module
Handles all memory persistence operations (MongoDB and file fallback)
"""

import logging
import json
import os
import time
import aiofiles
from typing import Dict, Any, Optional
from pymongo.database import Database

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Manages memory persistence to MongoDB and file storage"""

    def __init__(
        self, db: Optional[Database] = None, storage_path: Optional[str] = None
    ):
        self.db = db
        self.storage_path = storage_path
        # Collections
        self.conversations_collection = db.conversations if db is not None else None
        self.group_conversations_collection = (
            db.group_conversations if db is not None else None
        )
        self.conversation_summaries_collection = (
            db.conversation_summaries if db is not None else None
        )

    async def persist_memory(
        self, cache_key: str, memory_data: Dict[str, Any], is_group: bool = False
    ):
        """Persist memory to MongoDB storage"""
        try:
            if self.db is None:
                # Fallback to file storage if no MongoDB connection
                await self.persist_memory_file(cache_key, memory_data, is_group)
                return

            # Use MongoDB for persistence
            collection = (
                self.group_conversations_collection
                if is_group
                else self.conversations_collection
            )

            # Add timestamp and metadata
            memory_data.update(
                {
                    "cache_key": cache_key,
                    "is_group": is_group,
                    "last_updated": time.time(),
                }
            )

            # Update or insert conversation data
            collection.update_one(
                {"cache_key": cache_key}, {"$set": memory_data}, upsert=True
            )

            logger.info(
                f"Persisted {'group' if is_group else 'conversation'} memory to MongoDB for {cache_key}"
            )

        except Exception as e:
            logger.error(f"Error persisting memory to MongoDB: {e}")
            # Fallback to file storage on error
            await self.persist_memory_file(cache_key, memory_data, is_group)

    async def persist_memory_file(
        self, cache_key: str, memory_data: Dict[str, Any], is_group: bool = False
    ):
        """Fallback file-based persistence"""
        try:
            # Only attempt file persistence if storage_path is configured
            if self.storage_path is None:
                logger.debug(
                    f"No storage path configured, skipping file persistence for {cache_key}"
                )
                return

            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(
                self.storage_path, f"{'group_' if is_group else ''}{cache_key}.json"
            )

            # Add timestamp
            memory_data.update(
                {
                    "is_group": is_group,
                    "last_updated": time.time(),
                }
            )

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(memory_data, ensure_ascii=False, indent=2))

            logger.info(f"Persisted memory to file: {file_path}")

        except Exception as e:
            logger.error(f"Error persisting memory to file: {e}")

    async def load_memory(
        self, cache_key: str, is_group: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Load memory from MongoDB or file storage"""
        try:
            if self.db is not None:
                # Try MongoDB first
                collection = (
                    self.group_conversations_collection
                    if is_group
                    else self.conversations_collection
                )
                memory_data = collection.find_one({"cache_key": cache_key})

                if memory_data:
                    logger.info(
                        f"Loaded memory from MongoDB for {'group' if is_group else 'conversation'} {cache_key}"
                    )
                    return memory_data

            # Fallback to file storage
            return await self.load_memory_file(cache_key, is_group)

        except Exception as e:
            logger.error(f"Error loading memory from MongoDB: {e}")
            # Fallback to file storage on error
            return await self.load_memory_file(cache_key, is_group)

    async def load_memory_file(
        self, cache_key: str, is_group: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fallback file-based loading"""
        try:
            # Only attempt file loading if storage_path is configured
            if self.storage_path is None:
                logger.debug(
                    f"No storage path configured, skipping file loading for {cache_key}"
                )
                return None

            file_path = os.path.join(
                self.storage_path, f"{'group_' if is_group else ''}{cache_key}.json"
            )

            if not os.path.exists(file_path):
                return None

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                memory_data = json.loads(content)

            logger.info(
                f"Loaded memory from file for {'group' if is_group else 'conversation'} {cache_key}"
            )
            return memory_data

        except Exception as e:
            logger.error(f"Error loading memory from file: {e}")
            return None

    def ensure_indexes(self):
        """Ensure database indexes are created for better performance"""
        try:
            if self.conversations_collection is not None:
                # Index for conversation_id lookups
                self.conversations_collection.create_index("cache_key")
                self.conversations_collection.create_index("conversation_id")
                self.conversations_collection.create_index(
                    [("cache_key", 1), ("timestamp", -1)]
                )

            if self.group_conversations_collection is not None:
                # Indexes for group conversations
                self.group_conversations_collection.create_index("cache_key")
                self.group_conversations_collection.create_index("group_id")
                self.group_conversations_collection.create_index(
                    [("group_id", 1), ("timestamp", -1)]
                )

            if self.conversation_summaries_collection is not None:
                # Index for summary lookups
                self.conversation_summaries_collection.create_index("cache_key")
                self.conversation_summaries_collection.create_index("conversation_id")

            logger.info("Memory management database indexes ensured")

        except Exception as e:
            logger.error(f"Error creating memory management indexes: {e}")
