"""
Unit tests for model loading.
"""

import pytest
import torch


class TestModelLoading:
    """Tests for the SentimentModel singleton."""

    def test_model_loads_successfully(self):
        from src.models.model import sentiment_model

        assert sentiment_model.model is not None
        assert sentiment_model.tokenizer is not None

    def test_device_is_valid(self):
        from src.models.model import sentiment_model

        device = sentiment_model.get_device()
        assert device.type in ("cpu", "mps", "cuda")

    def test_model_is_in_eval_mode(self):
        from src.models.model import sentiment_model

        assert not sentiment_model.model.training

    def test_id2label_mapping_exists(self):
        from src.models.model import sentiment_model

        assert len(sentiment_model.id2label) > 0

    def test_tokenizer_encodes_korean(self):
        from src.models.model import sentiment_model

        tokens = sentiment_model.tokenizer("안녕하세요", return_tensors="pt")
        assert "input_ids" in tokens
        assert tokens["input_ids"].shape[0] == 1  # batch size 1
