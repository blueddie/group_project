import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# MS-COCO 데이터셋 클래스 정의
class CocoDataset:
    def __init__(self, json_file, image_folder):
        # MS-COCO 데이터셋 정보 로드
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder

    def __len__(self):
        # 데이터셋의 총 이미지 수 반환
        return len(self.data['images'])

    def __getitem__(self, idx):
        # 인덱스에 해당하는 이미지와 캡션 가져오기
        img_name = self.data['images'][idx]['file_name']
        img_id = self.data['images'][idx]['id']
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        # 이미지에 해당하는 캡션 가져오기
        captions = []
        for ann in self.data['annotations']:
            if ann['image_id'] == img_id:
                captions.append(ann['caption'])

        return image, captions

# 캡션 전처리 함수
def preprocess_caption(caption):
    # 표제어 추출 및 불용어 제거를 위한 준비
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # 소문자로 변환하고 토큰화
    tokens = word_tokenize(caption.lower())
    # 표제어 추출
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # 알파벳만 남기고 제거
    tokens = [token for token in tokens if token.isalpha()]
    # 불용어 제거
    tokens = [token for token in tokens if token not in stop_words]

    # 처리된 캡션 반환
    return ' '.join(tokens)

# 데이터셋과 데이터 제너레이터 준비
dataset = CocoDataset(json_file='C:\\group_project_data\\coco2017\\annotations\\captions_train2017.json', image_folder='C:\\group_project_data\\coco2017\\train2017\\train')

# 모델 정의, 손실 함수와 최적화 알고리즘 설정

# 모델 훈련

# 새 데이터에 대한 캡션 생성
