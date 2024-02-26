import os
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
import cv2

# print(model.info) # v8m, v5m6
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML 로 새모델 불러오고 가중치 땡겨오기
model = YOLO("yolov8n.pt")  # ""안에 모델 가중치 불러오기. n,s,m,l,x 순

result = model.predict('d:/_data/coco/archive/coco2017/train2017/000000000143.jpg', 
                        conf = 0.5, 
                        # show=True, 
                        # save_txt=True, 
                        max_det = 300, 
                        # visualize=True,
                        
                        )  # confidence 0.5이상만 박싱

print(result[0].shape)
# folder_path ='d:/_data/coco/archive/coco2017/train2017/'


class_names = model.names
class_id = result[0][2]
class_name = class_names[class_id]

# 결과 출력
print(f"클래스 이름: {class_name}")
print(f"클래스 ID: {class_id}")



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
