import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# YOLOv5 모델 불러오기
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지 불러오기
image_path = 'C:\\group_project_data\\coco2017\\train2017\\train\\000000000009.jpg'
image = Image.open(image_path)

# # 이미지를 YOLOv5 모델에 입력하여 객체 감지
results = model_yolo(image)

# # 결과 출력
# results.show()
# print(f"결과 \n {results}")

# 객체 감지 결과에서 클래스 레이블과 개수 추출
labels = results.names
counts = len(results.pred[0])  # 수정된 부분

print("객체 감지 결과:", counts, "개의 객체가 감지되었습니다.")

# 캡션 생성을 위한 Transformer 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model_caption = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# 감지된 객체 정보를 바탕으로 캡션 생성
caption = f"This image contains: "
for i in range(counts):
    label_id = results.pred[0][i][5].item()  # 클래스 정보는 5번째 인덱스에 있음
    label_name = labels[label_id]
    caption += f"{label_name}, "
caption = caption[:-2]  # 마지막 ", " 제거

print("캡션 생성 결과:", caption)
print("=====================================================================")

# Transformer 모델을 사용하여 캡션 생성
input_text = f"summarize: {caption}"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output_ids = model_caption.generate(input_ids)
caption_generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("YOLOv5로 감지된 객체 정보:", caption)
print("Transformer를 사용하여 생성된 캡션:", caption_generated)