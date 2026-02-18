# Inference logic for Korean Sentiment Analysis
import torch
import torch.nn.functional as F 

from .model import sentiment_model

def predict(text: str) -> dict:
    """
    Run sentiment prediction on the input text.
    Returns predicted label and confidence score.
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
        max_length=128
    )

    # Move tensors to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Get predicted class index
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()

    # If model has id2label mapping 
    if hasattr(model.config, "id2label"):
        label = model.config.id2label[predicted_class_id]
    else:
        label = str(predicted_class_id)

    return { 
        "label": label,
        "confidence": round(confidence, 4)
    }