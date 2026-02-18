"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Request body for the /predict endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Korean text to analyze for sentiment.",
        json_schema_extra={"examples": ["이 영화 정말 재미있어요!"]},
    )


class PredictionResponse(BaseModel):
    """Response body from the /predict endpoint."""

    label: str = Field(
        ...,
        description="Predicted emotion label in Korean.",
        json_schema_extra={"examples": ["기쁨(행복한)"]},
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1.",
        json_schema_extra={"examples": [0.9823]},
    )


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str
    model: str
    device: str
