from datetime import datetime, timedelta
from typing import Dict, List, Any
from .database import Database
import logging

class UserDataManager:
    def __init__(self):
        self.pool = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await Database.get_pool()

    async def initialize_user(self, user_id: int) -> None:
        """Initialize or update user data"""
        self.validate_user_id(user_id)
        async with self.pool.acquire() as conn:
            try:
                now = datetime.now()
                # Insert user with ON CONFLICT
                await conn.execute("""
                    INSERT INTO users (user_id, joined_date, last_active, messages_count)
                    VALUES ($1, $2, $2, 0)
                    ON CONFLICT (user_id) DO UPDATE 
                    SET last_active = $2
                """, user_id, now)

                # Initialize settings
                await conn.execute("""
                    INSERT INTO user_settings (user_id, markdown_enabled, code_suggestions)
                    VALUES ($1, true, true)
                    ON CONFLICT DO NOTHING
                """, user_id)

                # Initialize stats
                await conn.execute("""
                    INSERT INTO user_stats (user_id, total_messages, last_interaction)
                    VALUES ($1, 0, $2)
                    ON CONFLICT DO NOTHING
                """, user_id, now)

            except Exception as e:
                self.logger.error(f"Error initializing user {user_id}: {str(e)}")
                raise

    async def update_user_context(self, user_id: int, message: str, response: str) -> None:
        """Update user's conversation context"""
        self.validate_user_id(user_id)
        now = datetime.now()
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    # Insert message and response
                    await cur.execute("""
                        INSERT INTO user_contexts (user_id, role, content, created_at)
                        VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)
                    """, (user_id, "user", message, now, user_id, "assistant", response, now))

                    # Update user stats
                    await cur.execute("""
                        UPDATE users 
                        SET messages_count = messages_count + 1,
                            last_active = %s
                        WHERE user_id = %s
                    """, (now, user_id))

                except Exception as e:
                    self.logger.error(f"Error updating context for user {user_id}: {str(e)}")
                    raise

    async def get_user_context(self, user_id: int, limit: int = 20) -> List[Dict[str, str]]:
        """Get user's conversation history"""
        self.validate_user_id(user_id)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT role, content 
                    FROM user_contexts 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (user_id, limit))
                results = await cur.fetchall()
                return [{"role": role, "content": content} for role, content in results]

    async def get_user_settings(self, user_id: int) -> Dict[str, bool]:
        """Get user's settings"""
        self.validate_user_id(user_id)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT markdown_enabled, code_suggestions 
                    FROM user_settings 
                    WHERE user_id = %s
                """, (user_id,))
                result = await cur.fetchone()
                return {
                    'markdown_enabled': bool(result[0]) if result else True,
                    'code_suggestions': bool(result[1]) if result else True
                }

    async def toggle_setting(self, user_id: int, setting: str) -> None:
        """Toggle a user setting"""
        self.validate_user_id(user_id)
        valid_settings = ['markdown_enabled', 'code_suggestions']
        if setting not in valid_settings:
            raise ValueError(f"Invalid setting. Must be one of: {valid_settings}")

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    UPDATE user_settings 
                    SET {setting} = NOT {setting}
                    WHERE user_id = %s
                """, (user_id,))

    async def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        """Remove inactive users and their data"""
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    DELETE FROM users 
                    WHERE last_active < %s
                """, (threshold_date,))

    def validate_user_id(self, user_id: int) -> bool:
        """Validate user ID format"""
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("User ID must be a positive integer")
        return True