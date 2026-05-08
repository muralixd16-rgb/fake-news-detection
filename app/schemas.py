from pydantic import BaseModel

class ArticleRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float