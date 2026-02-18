# app/model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "nlp04/korean_sentiment_analysis_kcelectra"


class SentimentModel:
    def __init__(self):
        self.device = self._get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self):
        """
        Detect best available device.
        M1 Mac supports MPS.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device


# Singleton pattern — load once
sentiment_model = SentimentModel()
