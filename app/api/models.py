from pydantic import BaseModel
from typing import List, Dict, Optional


class Query(BaseModel):
    text: str
    context: Optional[Dict] = None


class SourceMetadata(BaseModel):
    content: str
    source: str
    chunk_index: int
    confidence: float


class Response(BaseModel):
    query: str
    response: str
    sources: List[SourceMetadata]
    timestamp: str


class Feedback(BaseModel):
    response_id: str
    rating: int
    comments: Optional[str] = None


class FactUpload(BaseModel):
    content: str
    source: str
    category: str