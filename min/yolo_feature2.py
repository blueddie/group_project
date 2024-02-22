import os
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
import cv2

# print(model.info) # v8m, v5m6
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML 로 새모델 불러오고 가중치 땡겨오기
model = YOLO("yolov8n.pt")  # ""안에 모델 가중치 불러오기. n,s,m,l,x 순

result = model.predict('d:/_data/coco/archive/coco2017/train2017/000000000143.jpg', conf = 0.5)  # confidence 0.5이상만 박싱

# folder_path ='d:/_data/coco/archive/coco2017/train2017/'
df = pd.DataFrame(result)
# print(df.shape())
## json
with open('df.json', 'w') as f:
    json.dump(result, f, indent=4)
## pandas
df.to_json('df.json', indent=4)
df.to_csv('df.csv', index=False)

# class_names = model.names
# class_id = result[0][2]
# class_name = class_names[class_id]

# # 결과 출력
# print(f"클래스 이름: {class_name}")
# print(f"클래스 ID: {class_id}")

# # 출력 정보 추출
# for detection in result:
#     # 객체 정보
#     x, y, w, h = detection[:4]
#     class_id = detection[5]
#     confidence = detection[6]

#     # 이미지 정보
#     image_width, image_height = model.model.image_size

#     # 클래스 이름
#     classes = model.model.names

#     # 출력
#     print(f"좌표: ({x}, {y}), ({x+w}, {y+h})")
#     print(f"클래스 ID: {class_id}")
#     print(f"신뢰도: {confidence}")
#     print(f"이미지 크기: ({image_width}, {image_height})")
#     print(f"클래스 이름: {classes[class_id]}")


# #######################################################################################################################
# class names :
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
# 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
# 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
# 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
# 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 
# 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
# 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
# 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
# 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
# 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
# 78: 'hair drier', 79: 'toothbrush'}