from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True)
    contexts = Column(JSON, default=[])
    settings = Column(JSON, default={
        'markdown_enabled': True,
        'code_suggestions': True
    })
    stats = Column(JSON, default={
        'messages': 0,
        'last_active': None,
        'joined_date': None
    })

class UserDataManager:
    def __init__(self):
        database_url = os.getenv('DATABASE_URL')
        if database_url and database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        # Add SSL configuration for production
        engine_args = {
            'echo': False,
            'pool_pre_ping': True,
        }
        
        # Add SSL requirements for production database
        if 'render.com' in database_url:
            engine_args['connect_args'] = {
                'sslmode': 'require'
            }
        
        self.engine = create_engine(database_url, **engine_args)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def validate_user_id(self, user_id: int) -> bool:
        if not isinstance(user_id, int):
            raise ValueError("User ID must be an integer")
        return True

    def initialize_user(self, user_id: int) -> None:
        self.validate_user_id(user_id)
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if not user:
            now = datetime.now().isoformat()
            new_user = User(
                user_id=user_id,
                contexts=[],
                settings={
                    'markdown_enabled': True,
                    'code_suggestions': True
                },
                stats={
                    'messages': 0,
                    'last_active': now,
                    'joined_date': now
                }
            )
            self.session.add(new_user)
            self.session.commit()

    def update_user_stats(self, user_id: int, message_count: int = 1) -> None:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if not user:
            self.initialize_user(user_id)
            user = self.session.query(User).filter_by(user_id=user_id).first()
        
        stats = user.stats
        stats['messages'] += message_count
        stats['last_active'] = datetime.now().isoformat()
        user.stats = stats
        self.session.commit()

    def get_user_context(self, user_id: int) -> list:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        return user.contexts if user else []

    def update_user_context(self, user_id: int, message: str, response: str) -> None:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if not user:
            self.initialize_user(user_id)
            user = self.session.query(User).filter_by(user_id=user_id).first()

        contexts = user.contexts
        contexts.append({"role": "user", "content": message})
        contexts.append({"role": "assistant", "content": response})
        
        if len(contexts) > 20:
            contexts = contexts[-20:]
        
        user.contexts = contexts
        self.update_user_stats(user_id)
        self.session.commit()

    def reset_user_data(self, user_id: int) -> None:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if user:
            user.contexts = []
            self.session.commit()

    def get_user_settings(self, user_id: int) -> dict:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        return user.settings if user else {
            'markdown_enabled': True,
            'code_suggestions': True
        }

    def toggle_setting(self, user_id: int, setting: str) -> None:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if not user:
            self.initialize_user(user_id)
            user = self.session.query(User).filter_by(user_id=user_id).first()
        
        settings = user.settings
        settings[setting] = not settings.get(setting, False)
        user.settings = settings
        self.session.commit()

    def update_user_settings(self, user_id: int, new_settings: dict) -> None:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        if not user:
            self.initialize_user(user_id)
            user = self.session.query(User).filter_by(user_id=user_id).first()
        
        settings = user.settings
        settings.update(new_settings)
        user.settings = settings
        self.session.commit()

    def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        current_time = datetime.now()
        users = self.session.query(User).all()
        
        for user in users:
            last_active = datetime.fromisoformat(user.stats['last_active'])
            if (current_time - last_active).days > days_threshold:
                self.session.delete(user)
        
        self.session.commit()

    def get_user_statistics(self, user_id: int) -> dict:
        user = self.session.query(User).filter_by(user_id=user_id).first()
        return user.stats if user else {}
