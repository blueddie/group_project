import os
import json
import torch
from PIL import Image


model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
image_path = "C:\\group_project_data\\coco2017\\train2017\\train\\000000000078.jpg"



def detect_objects(image_path, model):
    # 이미지 열기
    img = Image.open(image_path)

    # 객체 감지
    results = model(img)

    # 결과 출력
    print("객체 감지 결과:")
    for obj in results.xyxy[0]:
        label = int(obj[-1])
        conf = float(obj[-2])
        bbox = obj[:-2].tolist()
        print(f"클래스: {label}, 신뢰도: {conf}, 바운딩 박스: {bbox}")
    labels = [results.names[int(detection[5])] for detection in results.pred[0]]
    print(f"탐지된 라벨 : {labels}")

# 이미지 경로

# 객체 감지
detect_objects(image_path, model_yolo)
# =====================================================================================



