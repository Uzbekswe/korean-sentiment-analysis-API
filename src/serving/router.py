"""
API router for prediction endpoints.

Separating routes from app creation follows FastAPI best practices
and makes the codebase easier to test and extend.
"""

from fastapi import APIRouter, HTTPException

from src.models.inference import predict
from src.models.model import sentiment_model
from src.serving.schemas import TextRequest, PredictionResponse, HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint — confirms the API and model are running."""
    return HealthResponse(
        status="healthy",
        model=sentiment_model.model_name,
        device=str(sentiment_model.device),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_sentiment(request: TextRequest):
    """
    Predict the emotion of Korean text.

    Returns one of 11 Korean emotion labels with a confidence score.
    """
    try:
        result = predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
