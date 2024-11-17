from pydantic import BaseModel, Field
import uuid

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    user_id: str
    username: str
    first_name: str
    last_name: str 