import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
import PIL
import joblib
from joblib import dump, load
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle

# 기본 파일 위치
BASE_PATH = 'd:/_data/coco/archive/coco2017'                                # base 경로 설정

with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f:    # train.json 파일 열기. 열어서 변수 f에 할당
    data = json.load(f)                                                     # JSON 파일을 읽어서 파이썬 객체로 로드합니다. 
    data = data['annotations']                                              # 로드된 데이터에서 'annotations' 키의 값을 가져옴. 가져온 value를 변수 data에 할당

with open(f'{BASE_PATH}/annotations/captions_val2017.json', 'r') as f:      # val.json 파일 열기. 열어서 변수 f에 할당
    data2 = json.load(f)                                                    # JSON 파일을 읽어서 파이썬 객체로 로드합니다. 
    data2 = data2['annotations']  

img_cap_pairs = []                                                          # 이미지 이름과 caption을 매칭하여 담을 리스트 생성
for sample in data:                                                         # 데이터에서 각 샘플에 대해 반복합니다.
    img_name = '%012d.jpg' % sample['image_id']                             # 현재 샘플의 이미지 ID를 12자리 숫자로 포맷하여 이미지 파일 이름을 생성합니다.
    img_cap_pairs.append([img_name, sample['caption']])                     # 이미지 이름과 해당 이미지의 캡션을 쌍으로 묶어 리스트에 추가합니다

for sample in data2:                                                        
    img_name = '%012d.jpg' % sample['image_id']                             
    img_cap_pairs.append([img_name, sample['caption']])    


captions = pd.DataFrame(img_cap_pairs, columns = ['image', 'caption'])      # 이미지와 해당 이미지의 캡션 쌍을 사용하여 Pandas의 DataFrame을 생성합니다.
captions['image'] = captions['image'].apply(                                # 'image' 열의 각 요소에 대해 함수를 적용하여 수정합니다.
    lambda x: f'{BASE_PATH}/train2017/{x}'                                  # 이미지 파일의 경로를 수정하여 열에 적용합니다.
    # lambda x: f'{x}'                                  
)

print(captions.shape)
                                                    


def preprocess(text):
    text = text.lower()                                                     # 텍스트를 소문자로 변환합니다.
    text = re.sub(r'[^\w\s]', '', text)                                     # 특수 문자를 제거합니다.
    text = re.sub('\s+', ' ', text)                                         # 연속된 공백을 하나의 공백으로 대체합니다.
    text = text.strip()                                                     # 문자열 양쪽의 공백을 제거합니다.
    text = '[start] ' + text + ' [end]'                                     # 문장의 시작과 끝을 나타내는 토큰을 추가합니다.
    return text                                                             # 가공한 text 반환

captions['caption'] = captions['caption'].apply(preprocess)                 # captions 데이터프레임의 'caption' 열에 있는 모든 행에 preprocess 함수를 적용합니다.

random_row = captions.sample(1).iloc[0]                                     # 임의의 하나의 행을 random_row 변수에 할당                                                
im = Image.open(random_row.image)                                           # random_row 이미지에 접근
# im.show()                                                                 # 이미지 출력
# print(f"출력 이미지 정보 : {random_row}")     
# image      d:/_data/coco/archive/coco2017/train2017/00000...
# caption         [start] a man with a ball of some sort [end]     # 

MAX_LENGTH = 40
VOCABULARY_SIZE = 40000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 1

tokenizer = tf.keras.layers.TextVectorization(                              # "토큰(token)"은 텍스트를 작은 단위로 나누는 과정에서의 기본 단위를 의미.
    max_tokens=VOCABULARY_SIZE,                                             # max_tokens: 단어 집합의 크기를 결정. 즉, 텍스트에서 가장 빈도가 높은 상위 n개의 단어만을 사용하여 벡터화
    standardize=None,                                                       # 텍스트 표준화. 텍스트 데이터에 있는 다양한 변형이나 노이즈를 제거하고 일관된 형식으로 만들어주는 과정
    output_sequence_length=MAX_LENGTH                                       # 출력 시퀀스의 길이를 결정합니다. 모든 시퀀스를 동일한 길이로 만들기 위해 필요한 작업.
)

tokenizer.adapt(captions['caption'])
# print(captions['caption'])

print(tokenizer.vocabulary_size())                                          # 나누지 않은 train 전체    29080

# pickle.dump(tokenizer.get_vocabulary(), open(                             # 집합(vocabulary)을 파일로 저장
    # BASE_PATH + 'vocab_coco.file', 'wb'))  

word2idx = tf.keras.layers.StringLookup(
    mask_token="",                                                          # 마스크 처리를 하지 않는다? 
    vocabulary=tokenizer.get_vocabulary())                                  # StringLookup 레이어는 텍스트를 정수로 맵핑한다. tokenizer에 있는 단어사전에 맵핑된 정보를 쓴다.

# print(tokenizer.get_vocabulary())
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),  
    invert=True)                                                            # StringLookup 레이어는 invert=False가 default다. True로 한다면 정수를 입력 받아 해당하는 단어를 반환.

img_to_cap_vector = collections.defaultdict(list)                           # 이미지 이름을 키로 하고 캡션 리스트를 값으로 갖는 defaultdict 생성
for img, cap in zip(captions['image'], captions['caption']):                # captions 데이터프레임의 이미지 이름과 캡션에 대해 반복문 수행
    img_to_cap_vector[img].append(cap)                                      # 이미지 이름을 키로 캡션을 해당 이미지의 캡션 리스트에 추가
# print(img_to_cap_vector)                                                    # 
img_keys = list(img_to_cap_vector.keys())                                   # 이미지 이름들을 리스트로 저장
random.shuffle(img_keys)                                                    # 이미지 이름들을 무작위로 섞음
# print(img_to_cap_vector['d:/_data/coco/archive/coco2017/train2017/000000581245.jpg'])

slice_index = int(len(img_keys)*0.8)                                        # # slice_index를 계산하여 전체 이미지 개수의 80%에 해당하는 인덱스를 구합니다.
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index], 
                                          img_keys[slice_index:])

train_imgs = []                                                             # 이미지 이름을 저장할 리스트
train_captions = []                                                         # 이미지에 대한 캡션을 저장할 리스트
for imgt in img_name_train_keys:                                            #  훈련 데이터셋을 구성하는 이미지들에 대해서 반복
    capt_len = len(img_to_cap_vector[imgt])                                 # 현재 이미지에 대한 캡션의 개수를 계산
    train_imgs.extend([imgt] * capt_len)                                    # 현재 이미지를 훈련 데이터셋에 추가합니다. 캡션의 개수만큼 반복해서 추가
    train_captions.extend(img_to_cap_vector[imgt])                          # 해당 이미지에 대한 모든 캡션을 훈련 데이터셋에 추가

# print(train_imgs[:5])                                                       
# print(train_captions[:5])                                                   


val_imgs = []
val_captions = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector[imgv])
len(train_imgs), len(train_captions), len(val_imgs), len(val_captions)

def load_data(img_path, caption):
    img = tf.io.read_file(img_path)                                         # 이미지 파일을 바이트 문자열로 읽어옵니다.
    img = tf.io.decode_jpeg(img, channels=3)                                # 이미지를 RGB 채널의 텐서로 디코딩
    img = tf.keras.layers.Resizing(299, 299)(img)                           # 이미지를 299x299 픽셀 크기로 조정
    img = tf.keras.applications.inception_v3.preprocess_input(img)          # InceptionV3 모델에 입력으로 들어가기 전에 전처리를 수행 ( InceptionV3는 일반적으로 각 채널을 [-1, 1]범위로 정규화)  float32 
    caption = tokenizer(caption)                                            # 캡션을 토크나이징하여 벡터화
    return img, caption                                                     # 처리된 이미지와 벡터화된 캡션을 반환

train_dataset = tf.data.Dataset.from_tensor_slices(                         # 함수를 사용하여 데이터셋을 생성하고, 
    (train_imgs, train_captions))                                           # 각 이미지와 해당하는 캡션을 (train_imgs, train_captions) 튜플 형태로 저장합니다.

# for img, caption in train_dataset.take(5):
#     print("Image:", img)
#     print("Caption:", caption)
#     print()
    
train_dataset = train_dataset.map(
    load_data                                                               # 함수를 데이터셋의 각 요소에 적용합니다.
    , num_parallel_calls=tf.data.AUTOTUNE                                   # 데이터 전처리를 병렬로 처리
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)                                # 데이터셋을 셔플링, 이터셋을 배치로 만듭니다.

# for img, caption in train_dataset.take(1):
#     print(img[:5])
#     print(idx2word(caption[:5]))
    
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
