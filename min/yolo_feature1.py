import os
import numpy as np
from ultralytics import YOLO
import cv2

# print(model.info) # v8m, v5m6
model = YOLO("yolov8n.pt")  # ""안에 모델 가중치 불러오기. n,s,m,l,x 순
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML 로 새모델 불러오고 가중치 땡겨오기

result = model.predict('c:/123.mp4', conf = 0.5)  # confidence 0.5이상만 박싱
# folder_path ='c:/study/datasets/coco8/images/val/'
folder_path ='d:/_data/coco/archive/coco2017/train2017/'
# print(result)  # 탐지한 객체의 정보가 담김. (x,y,w,h), Class ID ? , confidence == 최종지표 mAP

# 단일 이미지 결과 보기
plots = result[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)  # 사용자의 키 입력 없이 자동으로 다음단계로 넘어가기 위한 코드. ex) 실시간 detection 등 연속된 것
cv2.destroyAllWindows() # 창이 닫힐때 까지 프로그램 대기상태. == 키입력을 하면 다음으로 넘어감. 마지막 사진일땐 끝남.

# for 문으로 폴더 안에 이미지들 전부 보기
# for filename in os.listdir(folder_path):
    # 폴더 안에 이미지 불러오기
    # image_path = os.path.join(folder_path, filename)
    # results = model.predict(image_path, conf = 0.5)
    
    # 시각화 
    # plots = results[0].plot()
    # cv2.imshow("plot", plots)    
    # # 키 입력 설정 (q = 종료, n = 다음 이미지)
    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     break
    # elif key == ord('n'):
    #     continue
# cv2.destroyAllWindows()
