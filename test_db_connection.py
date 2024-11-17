import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from src.services.user_data_manager import Base, User
import time

def test_database_connection():
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in .env file")
        return False
    
    print(f"🔄 Using database URL: {database_url.split('@')[1]}")  # Print only host part for security
        
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    max_retries = 3
    current_retry = 0
    
    while current_retry < max_retries:
        try:
            # Create engine with SSL settings
            print(f"\n🔄 Attempting to connect to database (attempt {current_retry + 1}/{max_retries})...")
            engine_args = {
                'echo': False,
                'pool_pre_ping': True,
                'connect_args': {
                    'sslmode': 'require',
                    'connect_timeout': 30
                }
            }
            engine = create_engine(database_url, **engine_args)
            
            # Test connection with a simple query
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
                print("✅ Successfully connected to database!")
                
            # Test table creation
            print("🔄 Attempting to create tables...")
            Base.metadata.create_all(engine)
            print("✅ Tables created successfully!")
            
            return True
            
        except SQLAlchemyError as e:
            print(f"❌ Database connection failed: {str(e)}")
            current_retry += 1
            if current_retry < max_retries:
                print(f"⏳ Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("❌ Max retries reached. Could not connect to database.")
                return False

if __name__ == "__main__":
    test_database_connection() 