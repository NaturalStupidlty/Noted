from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, Text, String

from sqlalchemy.ext.declarative import declarative_base
from src.config import settings

Base = declarative_base()


class Note(Base):
    __tablename__ = settings.INDEX_NAME

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    note_type = Column(String, default="note")  # Is a string to allow for future types
    topic = Column(String, default="uncategorized")
