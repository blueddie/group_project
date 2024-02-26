from transformers import pipeline
# 이미지 캡셔닝 파이프라인 생성
image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# 이미지 로드
# 예시: PIL 라이브러리를 사용하여 이미지를 로드합니다. 실제 사용 시에는 해당 경로를 이미지 파일 경로로 대체해야 합니다.
from PIL import Image
image = Image.open("C:\\_data\\project\\save_images\\frame_1.jpg")

# 이미지 캡셔닝 실행
results = image_captioning_pipeline(image)
print(results)











