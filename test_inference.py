# test_inference.py
from app.inference import predict

text = "이 영화 정말 재미있어요!"
result = predict(text)

print(result)