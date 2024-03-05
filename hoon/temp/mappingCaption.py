import json

# MS-COCO 캡션 파일 경로
caption_file_path = "C:\\group_project_data\\coco2017\\annotations\\captions_train2017.json"

# 캡션 파일 로드
with open(caption_file_path, "r") as f:
    captions_data = json.load(f)

# 이미지 ID와 캡션 매핑
image_id_to_captions = {}
for annotation in captions_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    if image_id not in image_id_to_captions:
        image_id_to_captions[image_id] = []
    image_id_to_captions[image_id].append(caption)

# 예시로 첫 번째 이미지의 캡션 출력
# first_image_id = list(image_id_to_captions.keys())[0]
# print("캡션 예시:")
# for caption in image_id_to_captions[first_image_id]:
#     print(caption)

#==============================================================
# 내가 원하는 id의 caption 모두 보기
# desired_image_id = 12345

# if desired_image_id in image_id_to_captions:
#     print(f"이미지 ID {desired_image_id}에 대한 캡션:")
#     for caption in image_id_to_captions[desired_image_id]:
#         print(caption)
# else:
#     print(f"이미지 ID {desired_image_id}에 대한 캡션이 존재하지 않습니다.")

print(image_id_to_captions.keys)