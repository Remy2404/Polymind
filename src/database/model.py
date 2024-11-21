from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True)
    settings = Column(JSON, default={
        'markdown_enabled': True,
        'code_suggestions': True
    })
    context = Column(JSON, default=[])
    messages_count = Column(Integer, default=0)
    last_active = Column(DateTime, default=datetime.utcnow)
    joined_date = Column(DateTime, default=datetime.utcnow)
