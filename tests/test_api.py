"""
Integration tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /"""

    def test_health_check_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_check_has_status_field(self):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_reports_model_name(self):
        response = client.get("/")
        data = response.json()
        assert "kcelectra" in data["model"].lower()


class TestPredictEndpoint:
    """Tests for POST /predict"""

    def test_predict_returns_200(self):
        response = client.post("/predict", json={"text": "이 영화 정말 재미있어요!"})
        assert response.status_code == 200

    def test_predict_returns_label_and_confidence(self):
        response = client.post("/predict", json={"text": "오늘 행복해요"})
        data = response.json()
        assert "label" in data
        assert "confidence" in data

    def test_predict_confidence_range(self):
        response = client.post("/predict", json={"text": "너무 슬퍼요"})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text_returns_422(self):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_text_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422
