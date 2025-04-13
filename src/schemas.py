from datetime import datetime
from pydantic import BaseModel


class NoteCreate(BaseModel):
    text: str


class NoteOut(BaseModel):
    id: int
    text: str
    created_at: datetime
    note_type: str
    topic: str

    class Config:
        from_attributes = True
