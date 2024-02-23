import os
import json
import numpy as np

captions_file = 'C:\\group_project_data\\coco2017\\annotations\\captions_train2017.json'


# def load_captions(captions_file):
#     with open(captions_file, 'r') as f:
#         captions_data = json.load(f)
#     return captions_data


# captions_data = load_captions(captions_file)
# print(captions_data)

def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    return captions_data

# captions_train2017.json 파일에서 캡션 정보만 추출하여 딕셔너리 형태로 저장
def extract_captions(captions_data):
    image_captions = {}
    for item in captions_data['annotations']:
        image_id = str(item['image_id'])  # 이미지 ID는 정수형이므로 문자열로 변환하여 사용
        caption = item['caption']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    return image_captions

# 사용 예시
captions_data = load_captions(captions_file)
image_captions = extract_captions(captions_data)

keys_list = list(image_captions.keys())
print(keys_list[0])  
print(image_captions[keys_list[0]])