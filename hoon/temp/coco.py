import os
import requests
from tqdm import tqdm

coco_dir = "c:/_data/coco/"

# 2017 훈련 데이터셋
train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
train_annotations_url = "http://images.cocodataset.org/annotations/annotations_train2017.zip"

# 2017 검증 데이터셋
val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
val_annotations_url = "http://images.cocodataset.org/annotations/annotations_val2017.zip"

# 2017 테스트 데이터셋
test_images_url = "http://images.cocodataset.org/zips/test2017.zip"
test_annotations_url = "http://images.cocodataset.org/annotations/annotations_test2017.zip"

# 다운로드 함수
def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        file_size = int(r.headers["Content-Length"])
        with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

# 훈련 데이터셋 다운로드
download_file(train_images_url, os.path.join(coco_dir, "train2017.zip"))
download_file(train_annotations_url, os.path.join(coco_dir, "annotations_train2017.zip"))

# 검증 데이터셋 다운로드
download_file(val_images_url, os.path.join(coco_dir, "val2017.zip"))
download_file(val_annotations_url, os.path.join(coco_dir, "annotations_val2017.zip"))

# 테스트 데이터셋 다운로드
download_file(test_images_url, os.path.join(coco_dir, "test2017.zip"))
download_file(test_annotations_url, os.path.join(coco_dir, "annotations_test2017.zip"))
