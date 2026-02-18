"""
FastAPI application factory.

Entry point for the Korean Sentiment Analysis API.
Run with: uvicorn src.serving.app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.serving.router import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Korean Sentiment Analysis API",
        description=(
            "Production-grade REST API for classifying Korean text into 11 emotion "
            "categories using a fine-tuned KcELECTRA transformer model."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware — allow Streamlit and other frontends
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    return application


app = create_app()
