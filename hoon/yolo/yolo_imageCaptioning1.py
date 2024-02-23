import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# YOLOv5 모델 불러오기
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 불러오기
image_path = 'C:\\group_project_data\\coco2017\\train2017\\train\\000000000036.jpg'
image = Image.open(image_path)

# # 이미지를 YOLOv5 모델에 입력하여 객체 감지
results = model_yolo(image)

# # 결과 출력
results.show()
# print(f"결과 \n {results}")

# 객체 감지 결과에서 클래스 레이블과 개수 추출
labels = results.names
counts = len(results.pred[0])  # 수정된 부분


print("객체 감지 결과:", counts, "개의 객체가 감지되었습니다.")

for detection in results.pred[0]:
    label_index = int(detection[5])  # 클래스 레이블의 인덱스 추출
    label = labels[label_index]
    print("감지된 객체:", label)