from PIL import Image
import os
import json

train_image_folder = 'C:\\group_project_data\\coco2017\\train2017\\'
test_image_folder = 'C:\\group_project_data\\coco2017\\test2017\\'
val_image_folder = 'C:\\group_project_data\\coco2017\\val2017\\'

# 이미지 파일 목록을 가져오는 함수
def get_image_files(folder):
    image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.jpg')]
    return image_files

# 훈련 이미지 파일 로드
train_image_files = get_image_files(train_image_folder)

# 테스트 이미지 파일 로드
test_image_files = get_image_files(test_image_folder)

# 검증 이미지 파일 로드
val_image_files = get_image_files(val_image_folder)

# 이미지를 로드하여 표시하는 함수
def show_images(image_files):
    for image_file in image_files:
        image = Image.open(image_file)
        image.show()  # 이미지 표시

# 훈련 이미지 표시
# show_images(train_image_files)

# 테스트 이미지 표시
# show_images(test_image_files)

# # 검증 이미지 표시
# show_images(val_image_files)

# print(len(train_image_files))   #118287
# print(len(test_image_files))   #40670
# print(len(val_image_files))   #5000

annotation_file_train = 'C:\\group_project_data\\coco2017\\annotations\\captions_train2017.json'
annotation_file_val = 'C:\\group_project_data\\coco2017\\annotations\\captions_val2017.json'

# JSON 파일에서 캡션 데이터를 로드하는 함수
def load_captions(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

# 캡션 데이터 로드
annotations_train = load_captions(annotation_file_train)
annotations_val = load_captions(annotation_file_val)

# 이미지 파일 이름과 캡션 정보를 추출하여 매칭하는 함수
def match_captions(image_files, annotations):
    image_captions = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        file_name = 'COCO_train2017_' + str(image_id).zfill(12) + '.jpg'  # 이미지 파일 이름
        if file_name in image_files:
            if file_name not in image_captions:
                image_captions[file_name] = []
            image_captions[file_name].append(caption)
    return image_captions

# 이미지 파일과 캡션 정보를 매칭
train_image_captions = match_captions(train_image_files, annotations_train)
val_image_captions = match_captions(val_image_files, annotations_val)

