import asyncio
import asyncpg
from dotenv import load_dotenv
import os
import logging

load_dotenv()

async def init_database():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        
        # Read SQL file
        with open('src/database/init.sql', 'r') as file:
            sql = file.read()
            
        # Execute SQL
        await conn.execute(sql)
        logging.info("Database initialized successfully")
        
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_database()) 