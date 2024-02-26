import os
import json
import torch
from PIL import Image


model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print(f"모델 로드 성공")

def get_labels_from_yolo(image_path):
    # YOLO 모델에 이미지 입력하여 객체 감지 결과 얻기
    results = model_yolo(Image.open(image_path))

    # 객체 감지 결과에서 클래스 레이블 추출
    labels = [results.names[int(detection[5])] for detection in results.pred[0]]

    return labels

# 캡션 정보를 로드하는 함수
def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    return captions_data





# 이미지 파일명을 키로 하고 라벨 및 캡션을 값으로 가지는 딕셔너리 생성
def create_data(images_dir, captions_data):
    data = {}
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):  # 이미지 파일인 경우에만 처리
            image_id = filename.split('.')[0]  # 파일명에서 확장자를 제거하여 이미지 ID 추출
            image_path = os.path.join(images_dir, filename)
            labels = get_labels_from_yolo(image_path)  # YOLO를 사용하여 이미지에서 라벨 추출
            print(f"{filename} 라벨 추출 성공")
            print(labels)
            if image_id in captions_data:  # 캡션 정보가 있는 경우에만 처리
                captions = captions_data[image_id]  # 이미지 ID에 해당하는 캡션 정보 가져오기
                data[filename] = {"labels": labels, "captions": captions}  # 이미지 파일명을 키로 사용하여 데이터 구성
                print(f"파일명 : {filename} 완료")
    return data

#=======================================================================================================
images_dir = 'C:\\_data\\project\\coco\\coco2017\\train2017\\'

captions_file = 'C:\\_data\\project\\coco\\coco2017\\annotations\\captions_train2017.json'  # captions_train2017.json 파일의 경로
captions_data = load_captions(captions_file)

# dict_keys 객체를 리스트로 변환
keys = list(captions_data.keys())
print(keys)

# 이제 첫 번째 키를 사용하여 첫 번째 아이템에 접근할 수 있음



print(captions_data[keys[3]])
# print(f"json 로드 성공")
# data = create_data(images_dir, captions_data)




# =============================파일 저장=============================
# 저장할 디렉토리 경로
# output_dir = 'C:\\group_project_data\\makeData'

# # 저장할 파일명
# output_filename = 'imageLabelCaption.json'

# # 파일의 전체 경로
# output_file = os.path.join(output_dir, output_filename)

# # data 딕셔너리를 JSON 파일로 저장
# with open(output_file, 'w') as f:
#     json.dump(data, f)

# print(f"데이터가 {output_file} 파일로 성공적으로 저장되었습니다.")