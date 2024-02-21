import cv2
from transformers import pipeline

# 이미지 캡셔닝 파이프라인 초기화
image_captioning = pipeline(task="image-captioning", model="Model_Name")

# 이미지 파일로부터 설명 생성
image_path = "frame1.jpg" # 앞서 추출한 프레임 중 하나
result = image_captioning(image_path)
caption = result[0]["caption"]

print("Generated caption:", caption)









