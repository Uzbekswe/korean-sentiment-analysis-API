"""
Unit tests for the inference pipeline.
"""

import pytest


class TestInference:
    """Tests for the predict() function."""

    def test_predict_returns_label_and_confidence(self):
        from src.models.inference import predict

        result = predict("이 영화 정말 재미있어요!")
        assert "label" in result
        assert "confidence" in result

    def test_confidence_is_between_0_and_1(self):
        from src.models.inference import predict

        result = predict("오늘 너무 슬퍼요")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_label_is_non_empty_string(self):
        from src.models.inference import predict

        result = predict("화가 나요!")
        assert isinstance(result["label"], str)
        assert len(result["label"]) > 0

    def test_predict_handles_long_text(self):
        from src.models.inference import predict

        long_text = "좋아요 " * 200  # exceeds max_length, should be truncated
        result = predict(long_text)
        assert "label" in result

    def test_predict_handles_single_character(self):
        from src.models.inference import predict

        result = predict("아")
        assert "label" in result
