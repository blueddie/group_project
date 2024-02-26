import os
import json
import torch
from PIL import Image
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils import pad_sequences


model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print(f"모델 로드 성공")

def get_labels_from_yolo(image_path):
    # YOLO 모델에 이미지 입력하여 객체 감지 결과 얻기
    results = model_yolo(Image.open(image_path))

    # 객체 감지 결과에서 클래스 레이블 추출
    labels = [results.names[int(detection[5])] for detection in results.pred[0]]

    return labels

def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    return captions_data



# 캡션 정보를 로드하는 함수
def create_data(images_dir, captions_data, num_images=10):
    data = {}
    image_count = 0
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):  # 이미지 파일인 경우에만 처리
            image_id = int(filename.split('.')[0])  # 파일명에서 확장자를 제거하여 이미지 ID 추출
            image_path = os.path.join(images_dir, filename)
            labels = get_labels_from_yolo(image_path)  # YOLO를 사용하여 이미지에서 라벨 추출
            
            # 해당 이미지의 모든 캡션을 저장할 리스트 생성
            captions_list = []
            
            for annotation in captions_data['annotations']:
                if annotation['image_id'] == image_id:
                    captions = annotation['caption']  # 이미지에 대한 캡션 정보 가져오기
                    captions_list.append(captions)  # 캡션을 리스트에 추가
                    
            # 해당 이미지에 대한 라벨과 모든 캡션을 데이터에 추가
            data[filename] = {"labels": labels, "captions": captions_list}
            image_count += 1

            if image_count >= num_images:
                break

    return data

#=======================================================================================================
images_dir = 'C:\\_data\\project\\coco\\coco2017\\train2017'

captions_file = 'C:\\_data\\project\\coco\\coco2017\\annotations\\captions_train2017.json'  # captions_train2017.json 파일의 경로
captions_data = load_captions(captions_file)

# print(f"json 로드 성공")
data = create_data(images_dir, captions_data, num_images=10)
# ================================================================
# for annotation in captions_data['annotations']:
#     print(annotation)
#     # print(annotation['caption'])
#     print("===================")
#     break

# =============================파일 저장=============================
# 저장할 디렉토리 경로
output_dir = 'C:\\_data\\project'

# # 저장할 파일명
output_filename = 'imageLabelCaption3.json'

# # 파일의 전체 경로
output_file = os.path.join(output_dir, output_filename)

# # data 딕셔너리를 JSON 파일로 저장
with open(output_file, 'w') as f:
    json.dump(data, f)

print(f"데이터가 {output_file} 파일로 성공적으로 저장되었습니다.")









