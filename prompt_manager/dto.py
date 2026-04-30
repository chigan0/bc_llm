from pydantic import BaseModel


class CacheEntry(BaseModel):
    mtime: float
    content: str
