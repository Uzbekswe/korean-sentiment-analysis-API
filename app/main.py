# Entry point for the Korean Sentiment Analysis API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .inference import predict

app = FastAPI(
    title="Korean Sentiment Analysis API",
    description="API for performing sentiment analysis on Korean text using a pre-trained transformer model.",
    version="1.0.0"
)

# Request schema
class TextRequest(BaseModel):
    text: str

# Response schema
class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: TextRequest):
    try:
        result = predict(request.text)
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))