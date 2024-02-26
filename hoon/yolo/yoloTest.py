import torch
from transformers import AutoModelForSeq2SeqLM

# YOLO 모델 로드
yolo_model = torch.load("yolo_model.pt")