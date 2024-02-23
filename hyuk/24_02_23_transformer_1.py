import tensorflow as tf
from keras.applications import InceptionV3
from keras.models import Model
import re
import os

# 데이터셋 경로 설정
dataset_path = "/home/angeligareta/Downloads/Resources"
dataset_images_path = os.path.join(dataset_path, "Images/")

# 이미지 크기 및 분할 비율 설정
img_height = 180
img_width = 180
validation_split = 0.2

# 이미지 특성 추출 모델 생성 함수 정의
def get_encoder():
    image_model = InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = Model(new_input, hidden_layer)
    return image_features_extract_model

# 캡션 전처리 함수 정의
def get_preprocessed_caption(caption):    
    caption = re.sub(r'\s+', ' ', caption)          
    caption = caption.strip()                       
    caption = "<start> " + caption + " <end>"
    return caption

# 이미지 파일 이름과 캡션 정보를 저장할 딕셔너리 초기화
images_captions_dict = {}

# 캡션 파일 읽어와서 딕셔너리에 저장
with open(os.path.join(dataset_path, "captions.txt"), "r") as dataset_info:
    next(dataset_info) # 헤더 건너뛰기

    # 최대 4000개의 캡션만 사용
    for info_raw in list(dataset_info)[:4000]:
        info = info_raw.split(",")
        image_filename = info[0]
        caption = get_preprocessed_caption(info[1])

        if image_filename not in images_captions_dict.keys():
            images_captions_dict[image_filename] = [caption]
        else:
            images_captions_dict[image_filename].append(caption)

# 영상 데이터의 프레임 이미지 디렉토리 설정
video_frames_dir =  "c:\\_data\\project\\save_images3\\"

# 프레임 이미지 파일 리스트를 가져와서 딕셔너리에 추가
frame_files = os.listdir(video_frames_dir)
for frame_file in frame_files:
    image_filename = os.path.basename(frame_file)
    caption = ""  # 프레임 이미지에 대한 캡션 정보가 있다면 적절한 방식으로 생성
    
    if image_filename not in images_captions_dict:
        images_captions_dict[image_filename] = [caption]
    else:
        images_captions_dict[image_filename].append(caption)           # # 이미지 파일 이름이 딕셔너리의 키로 이미 존재하는 경우에는 해당 키에 대한 값을 가져와서 캡션을 추가










