# 🇰🇷 Korean Sentiment Analysis API

Production-grade REST API and web UI for classifying Korean text into **11 emotion categories** using a fine-tuned [KcELECTRA](https://huggingface.co/nlp04/korean_sentiment_analysis_kcelectra) transformer model.

> **Live Demo:** [Streamlit App](https://uzbekswe-korean-sentiment-analysis-api.streamlit.app)

---

## Architecture

```
┌─────────────────┐     HTTP POST      ┌──────────────────────┐
│  Streamlit UI   │ ──────────────────► │    FastAPI Server     │
│  (streamlit_app │     /predict        │  src/serving/app.py   │
│   .py)          │ ◄────────────────── │                       │
└─────────────────┘     JSON response   └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │   Inference Engine    │
                                        │ src/models/inference  │
                                        │         .py           │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │   KcELECTRA Model    │
                                        │ (HuggingFace)        │
                                        │ 11 emotion classes   │
                                        └──────────────────────┘
```

### Inference Pipeline

```
Korean text → Tokenize → Feed to model → Raw scores (logits)
→ Softmax → Probabilities → argmax → Emotion label + confidence
```

---

## Emotion Labels

| ID | Korean Label | English |
|----|-------------|---------|
| 0 | 기쁨(행복한) | Joy (Happy) |
| 1 | 슬픔 | Sadness |
| 2 | 분노 | Anger |
| 3 | 불안 | Anxiety |
| 4 | 상처(배신당한) | Hurt (Betrayed) |
| 5 | 당황 | Embarrassment |
| 6 | 기쁨 | Joy |
| 7 | 놀람 | Surprise |
| 8 | 혐오 | Disgust |
| 9 | 공포 | Fear |
| 10 | 중립 | Neutral |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Uzbekswe/korean-sentiment-analysis-API.git
cd korean-sentiment-analysis-API

python -m venv .venv && source .venv/bin/activate
pip install ".[dev]"
```

### 2. Run the API

```bash
make serve
# or: uvicorn src.serving.app:app --reload
```

### 3. Test it

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "이 영화 정말 재미있어요!"}'
```

**Response:**
```json
{
  "label": "기쁨(행복한)",
  "confidence": 0.9823
}
```

### 4. Run the Streamlit UI

```bash
make streamlit
# or: streamlit run streamlit_app.py
```

### 5. Run with Docker

```bash
make docker-up
# or: docker compose -f docker/docker-compose.yml up --build
```

---

## Project Structure

```
korean-sentiment-analysis-API/
├── src/
│   ├── models/              # Model loading, config, inference
│   │   ├── model.py         # SentimentModel singleton (loads KcELECTRA)
│   │   ├── inference.py     # predict() — tokenize → model → softmax → label
│   │   └── config.py        # Reads configs/model_config.yaml
│   ├── serving/             # FastAPI application
│   │   ├── app.py           # App factory with CORS middleware
│   │   ├── router.py        # GET / (health) + POST /predict endpoints
│   │   └── schemas.py       # Pydantic request/response models
│   └── monitoring/          # Prediction logging
│       └── logger.py        # JSONL prediction logger
├── tests/                   # pytest test suite
│   ├── test_model.py        # Model loading tests
│   ├── test_inference.py    # Inference pipeline tests
│   └── test_api.py          # FastAPI integration tests
├── configs/
│   └── model_config.yaml    # Model hyperparameters (no hardcoded values)
├── docker/
│   ├── Dockerfile           # Multi-stage production build
│   └── docker-compose.yml   # One-command deployment
├── .github/workflows/
│   └── ci.yml               # Lint → Test → Docker build pipeline
├── notebooks/               # Exploratory analysis only
├── streamlit_app.py         # Streamlit web UI (self-contained)
├── .streamlit/config.toml   # Streamlit theme settings
├── pyproject.toml           # Dependencies & tool config
├── Makefile                 # make serve, make test, make docker-build
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # You are here
```

---

## Development

```bash
make dev          # Install all dependencies
make test         # Run tests
make lint         # Run ruff linter
make format       # Auto-format code
make clean        # Remove caches
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | KcELECTRA (fine-tuned, 11 emotions) |
| Inference | PyTorch + HuggingFace Transformers |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Config | YAML (no hardcoded values) |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker (multi-stage) |
| Linting | Ruff |
| Dependency Mgmt | pyproject.toml |

---

## API Docs

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## License

MIT
