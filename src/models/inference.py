"""
Inference logic for Korean Sentiment Analysis.

Takes raw Korean text, tokenizes it, runs the model,
and returns the predicted emotion label with confidence score.
"""

import torch
import torch.nn.functional as F

from src.models.model import sentiment_model


def predict(text: str) -> dict:
    """
    Run sentiment prediction on Korean input text.

    Args:
        text: Korean text string to classify.

    Returns:
        dict with keys:
            - label (str): Predicted emotion label (e.g. "기쁨(행복한)")
            - confidence (float): Probability score between 0 and 1
    """
    tokenizer = sentiment_model.get_tokenizer()
    model = sentiment_model.get_model()
    device = sentiment_model.get_device()

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=sentiment_model.max_length,
    )

    # Move tensors to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()

    # Map class ID to human-readable label
    if hasattr(model.config, "id2label"):
        label = model.config.id2label[predicted_class_id]
    else:
        label = str(predicted_class_id)

    return {
        "label": label,
        "confidence": round(confidence, 4),
    }
