"""
Basic prediction logging for monitoring.

Logs every prediction to a JSONL file for later analysis
(drift detection, accuracy tracking, debugging).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"


def log_prediction(text: str, label: str, confidence: float) -> None:
    """Append a prediction record to the JSONL log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text[:200],  # truncate for storage
        "label": label,
        "confidence": confidence,
    }
    try:
        with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")
