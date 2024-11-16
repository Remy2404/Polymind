import asyncpg
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

class Database:
    _pool = None

    @classmethod
    async def get_pool(cls):
        if cls._pool is None:
            retries = 5
            while retries > 0:
                try:
                    cls._pool = await asyncpg.create_pool(
                        os.getenv('DATABASE_URL'),
                        min_size=1,
                        max_size=10,
                        ssl='require' if 'render' in os.getenv('DATABASE_URL', '') else None
                    )
                    logger.info("Database connection pool created successfully")
                    break
                except Exception as e:
                    retries -= 1
                    if retries == 0:
                        logger.error(f"Failed to create database pool after 5 attempts: {str(e)}")
                        raise
                    logger.warning(f"Failed to connect to database, retrying... ({retries} attempts left)")
                    await asyncio.sleep(5)
        return cls._pool

    @classmethod
    async def close(cls):
        if cls._pool:
            await cls._pool.close()
            logger.info("Database connection closed")