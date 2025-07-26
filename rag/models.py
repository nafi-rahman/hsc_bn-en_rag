from pydantic import BaseModel
from typing import Dict, List, Any

class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]

class Answer(BaseModel):
    text: str
    sources: List[Dict[str, Any]]