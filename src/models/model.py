"""
Model loader for Korean Sentiment Analysis.

Loads the pre-trained KcELECTRA model and tokenizer as a singleton
so the ~500MB model is only loaded once into memory.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.config import get_model_config


class SentimentModel:
    """Singleton wrapper around the KcELECTRA sentiment model."""

    def __init__(self):
        config = get_model_config()
        self.model_name = config["model_name"]
        self.max_length = config["max_length"]
        self.device = self._get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _get_device() -> torch.device:
        """Detect best available device: CUDA > MPS (Apple Silicon) > CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def id2label(self) -> dict:
        """Return the model's id-to-label mapping."""
        return getattr(self.model.config, "id2label", {})

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device


# Singleton — instantiated once at import time
sentiment_model = SentimentModel()
