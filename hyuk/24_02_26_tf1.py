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
# from tqdm.auto import tqdm
import pickle



BASE_PATH = 'd:/_data/coco/archive/coco2017'
with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f:
    data = json.load(f)                         # 딕셔너리 형태
    data = data['annotations']




# Total images: 118287

# print(data)

img_cap_pairs = []  

for sample in data:
    img_name = '%012d.jpg' % sample['image_id']         # sample 딕셔너리에서 image_id 값을 가져와서 이를 12자리 숫자 형태의 문자열로 포맷
    img_cap_pairs.append([img_name, sample['caption']])     #  # 결국 img_cap_pairs 에는 이미지파일 이름과 대응하는 캡션이 들어있는 리스트들이 모인다





captions = pd.DataFrame(img_cap_pairs, columns = ['image', 'caption'])          # image랑 caption 2개의 열이 있는 데이터프레임 생성
captions['image'] = captions['image'].apply(lambda x: f'{BASE_PATH}/train2017/{x}') 

# print(captions.shape)


# captions.to_csv(BASE_PATH +'train.csv', index=False )





captions = captions.sample(70000)           # 데이터셋의 크기를 줄이고, 하스에 사용할 데이터의 다영성을 유지하기 위헤 7만개 랜덤으로 선택
captions = captions.reset_index(drop=True)  # 샘플링 후 인덱스를 리셋함
# print(captions.head())
# print(captions.info())
# print(captions.describe())

def preprocess(text):                       # 캡션의 전처리 
    text = text.lower()                     # 소문자
    text = re.sub(r'[^\w\s]', '', text)     # 단어가 아닌애들 제거
    text = re.sub('\s+', ' ', text)         # 공백을 하나로 줄이기
    text = text.strip()                     # 양끝의 공백을 제거
    text = '[start]' + text + ' [end]'      # 각 캡션의 시작과 끝에 특수토큰 추가
    return text

captions['caption'] = captions['caption'].apply(preprocess)     # 모든 캡션에 preprocess함수 적용






random_row = captions.sample(1).iloc[0]     # 샘플 하나를 무작위로 선택 후 첫번째 행을 반환
# print(random_row.caption)  
# print()
# im = Image.open(random_row.image)
# im.show()

MAX_LENGTH = 40             # 시퀀스의 최대 길이
VOCABULARY_SIZE = 15000     # 모델이 처리할 수 있는 어휘의 최대 크기
BATCH_SIZE = 64
BUFFER_SIZE = 1000          # 데이터를 섞을떄 사용되는 버퍼의 크기
EMBEDDING_DIM = 512         # 차원 수, 단어의 의미를 벡터공간에 저장
UNITS = 512                 # 모델의 숨겨진 층에서 사용되는 유닛(뉴런)의 수를 저으이
EPOCHS = 30

tokenizer = tf.keras.layers.TextVectorization(          # TextVectorization : 텍스트 데이터를 수치형 벡터 데이터로 변환하는 과정을 자동화
                                                        #                     초기 데이터 전처리 단계에서 필요한 텍스트 처리를 효율적으로 수행
    max_tokens=VOCABULARY_SIZE,             #  사용할 최대 토큰 수(여기서는 어휘 크기)
    standardize=None,                       # 텍스트를 전처리하는 방법(여기서는 전처리를 수행하지 않음)
output_sequence_length=MAX_LENGTH           # 출력 시퀀스의 길이를 정의
)
tokenizer.adapt(captions['caption'])    # 토큰화를 수행하고 어휘를 구축, 토큰화 객체는 데이터를 분석하여 모델에 필요한 어휘를 생성
print(tokenizer.vocabulary_size())  # 누를때마다 바뀜 아마 random_row 때문인거같음

pickle.dump(tokenizer.get_vocabulary(), open('vocab_coco.file', 'wb'))      
# TextVectorization 층에 의해 구축된 어휘 목록을 토큰화된 텍스트 데이터에서 추출된 고유한 단어들의 리스트

word2idx = tf.keras.layers.StringLookup(    #  모델의 어휘 내 각 단어를 해당하는 정수 인덱스로 변환하는 매핑을 생성
    mask_token = "",                        #  무시해야 할 토큰을 지정
    vocabulary = tokenizer.get_vocabulary()) #  StringLookup 층에 사용할 어휘를 지정합니다. 이전 단계에서 TextVectorization 층을 사용하여 생성된 어휘 목록을 이용

idx2word= tf.keras.layers.StringLookup(
    mask_token = "",
    vocabulary = tokenizer.get_vocabulary(),
    invert = True)  #  invert=True 매개변수를 통해 정수 인덱스를 문자열(단어)로 매핑하는 역방향 매핑을 생성
    # 모델 출력으로부터 얻은 정수 인덱스를 다시 단어로 변환

# print(word2idx)
# print(idx2word)
import collections      

img_to_cap_vector = collections.defaultdict(list)       # defaultdict를 사용하여 img_to_cap_vector라는 딕셔너리를 생성
#  파일 이름(img)을 키로 하고, 해당 이미지에 대한 캡션들(cap)을 값으로 하는 리스트를 관리, defaultdict(list)는 존재하지 않는 키에 대해 자동으로 빈 리스트를 기본값으로 설정
for img, cap in zip (captions['image'], captions['caption']):   # zip 함수를 사용하여 이미지 이름과 캡션을 짝지어 반복 처리
    img_to_cap_vector[img].append(cap)  # 각 이미지 이름을 키로 사용하여 img_to_cap_vector 딕셔너리에 캡션을 추가, 이미지 이름이 이미 키로 존재하면, 해당 캡션은 리스트에 추가, 
# 존재하지 않는 경우, 자동으로 빈 리스트가 생성되고 캡션은 이 리스트에 추가
    
img_keys = list(img_to_cap_vector.keys())   # img_to_cap_vector 딕셔너리의 키(이미지 이름)들을 추출하여 리스트로 변환, 이 리스트는 모든 고유 이미지 파일 이름을 포함.
random.shuffle(img_keys)        # img_keys 리스트의 순서를 무작위로 섞음, 데이터셋의 훈련세트와 검증세트를 나눌때 데이터의 분포가 고르게 도와줌


print(captions['caption'].shape)



slice_index = int(len(img_keys)*0.8)        # img_keys 리스트 길이의 80퍼에 해당하는 위치를 계산 즉, 훈련세트와 검증세트를 나누는 기준점 
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index], img_keys[slice_index:])    # img_keys 리스트를 두 부분으로 나눔
# 분할된 두 리스트는 각각 img_name_train_keys와 img_name_val_keys 변수에 할당

train_imgs = []
train_captions = []
for imgt in img_name_train_keys:                # 훈련 데이터셋에 사용될  이미지 이름을 포함 
    capt_len = len(img_to_cap_vector[imgt])     # 해당 이미지에 연결된 캡션들의 리스트를 가져옴, capt_len : 해당 이미지에 대한 캡션 수 나타냄
    train_imgs.extend([imgt] * capt_len)        # train_imgs 리스트에 현재 이미지 이름을 capt_len만큼 반복하여 추가, 이는 각 캡션에 해당하는 이미지 이름을 동일하게 유지하기 위함
    train_captions.extend(img_to_cap_vector[imgt])      # train_captions 리스트에 해당 이미지의 모든 캡션을 추가합니다.
    
val_imgs = []
val_captions = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector[imgv])    
len(train_imgs), len(train_captions), len(val_imgs), len(val_captions)

# print(len(train_imgs), len(train_captions), len(val_imgs), len(val_captions))
# 55897 55897 14103 14103

def load_data(img_path, caption):   # load_data 함수는 주어진 img_path와 caption을 입력으로 받아, 이미지를 로드하고 전처리 후 해당 이미지와 토큰화된 캡션을 반환
    img = tf.io.read_file(img_path)                                     # 파일의 내용을 바이트 형태로 반환
    img = tf.io.decode_jpeg(img, channels=3)                            # 읽어온 이미지 파일(바이트 데이터)를 JPEG 포맷으로 디코드(해석), channels=3 매개변수는 이미지가 RGB 채널을 가진 컬러 이미지임을 나타냄
    img = tf.keras.layers.Resizing(299, 299)(img)                       # InceptionV3 모델과 같은 특정 모델에서 요구하는 입력 크기 (299, 299)
    img = tf.keras.applications.inception_v3.preprocess_input(img)   # 이 함수를 이용하여 이미지데이터를 전처리, InceptionV3 모델에 적합한 형태로 이미지 데이터를 스케일링하고 정규화
    caption = tokenizer(caption)                                        # 이전에 정의된 tokenizer 객체를 사용하여 입력 캡션을 토큰화, 이 과정에서 캡션은 모델이 처리할 수 있는 수치 데이터(토큰의 시퀀스)로 변환
    return img, caption                                                 # 전처리된 이미지와 토큰화된 캡션을 반환

train_dataset = tf.data.Dataset.from_tensor_slices(     # train_imgs와 train_captions 배열을 사용하여 TensorFlow의 데이터셋 객체를 생성,
    (train_imgs, train_captions))                       # 이 함수는 데이터셋을 각각의 이미지-캡션 쌍으로 분할 -> 튜플로 구성해줌

train_dataset = train_dataset.map(                      # load_data 함수를 사용하여 데이터셋을 변환, 데이터를 로드하고 전처리하는 작업을 포함할 수 있음,
    load_data, num_parallel_calls=tf.data.AUTOTUNE      # num_parallel_calls=tf.data.AUTOTUNE : 데이터 변환을 병렬로 처리하기 위해 TensorFlow에게 최적화를 위임
    ).shuffle(BUFFER_SIZE                               #  BUFFER_SIZE는 섞을 버퍼의 크기를 나타내며, 큰 데이터셋을 다룰 때 메모리를 효율적으로 사용하기 위해 사용
              ).batch(BATCH_SIZE)                       # 데이터셋을 배치로 묶는다, BATCH_SIZE는 각 배치에 포함될 샘플의 개수를 나타냄

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# for img_path, caption in train_dataset.take(1):
#     print(img_path)     # (64, 299, 299, 3)
#     print(caption)      # (64, 40)

image_augmentation = tf.keras.Sequential(
    [   tf.keras.layers.RandomFlip("horizontal"),       # 이미지를 수평 방향으로 무작위로 뒤집는(flipping) 작업을 수행
        tf.keras.layers.RandomRotation(0.2),            # 이미지를 무작위로 회전시키는 작업을 수행
        tf.keras.layers.RandomContrast(0.3),            # 이미지의 대비를 무작위로 조절하는 작업을 수행
     ]
)

class TransformerEncoderLayer(tf.keras.layer):          # TransformerEncoderLayer 클래스는 tf.keras.layers.Layer를 상속받아 정의된다. 이를 통해 사용자 정의 층(custom layer)을 만들 수 있다
    
    def __init__(self, embed_dim, num_heads):           # 클래스의 생성자(__init__)에서는 임베딩 차원(embed_dim)과 멀티 헤드 어텐션에서의 헤드 수(num_heads)를 매개변수로 받습니다.
        super().__init__()                                # super().__init__()를 호출하여 부모 클래스(tf.keras.layers.Layer)의 생성자를 초기화합니다.
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()    # 두 개의 층 정규화(layer normalization) 층을 정의
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()    # 층 정규화는 입력 데이터의 평균과 분산을 사용하여 정규화를 수행하고, 모델의 학습을 안정화시키는 데 도움을 준다.
        self.attention = tf.keras.layers.MultiHeadAttention(        # 멀티 헤드 어텐션 층을 정의
            num_heads=num_heads, key_dim=embed_dim)                 # 이 층은 입력 데이터의 다른 부분에 대한 정보를 병렬로 처리할 수 있게 해주며, 입력 시퀀스 내의 다양한 위치에서 정보를 통합
        self.dense = tf.keras.layers.Dense(embed_dim, activation='relu')    # 포지션 와이즈 피드포워드 신경망을 위한 밀집(dense) 층을 정의합니다. 이 층은 각 위치에서 동일하게 적용되며, 비선형 변환을 제공
        

    def call(self, x, training):
        x = self.layer_norm_1(x)        # 입력 x에 대해 첫 번째 레이어 정규화를 수행, 각 입력의 특성(feature)들이 평균 0, 분산 1을 가지도록 정규화하는 과정. 학습 과정을 안정화시키고, 학습 속도를 향상 굿
        x = self.dense(x)               # 정규화된 입력 x를 밀집(dense) 층에 통과시킴. 이 밀집 층은 입력에 대해 가중치를 적용하고 활성화 함수(여기서는 'relu')를 통과시킨다,
                                        # 이 과정은 입력 데이터에 비선형 변환을 적용
        
        attn_output = self.attention(   # 멀티 헤드 어텐션 연산을 수행
            query=x,                    #  query, value, key 모두 자기 자신에 대한 어텐션(self-attention)을 의미
            value=x,
            key=x,
            attention_mask=None,        # 어텐션 마스크를 사용하지 않겠다는 것
            training=training           # 모델이 학습 모드인지 추론 모드인지를 나타내며, 이는 드롭아웃 같은 일부 연산이 학습 시와 추론 시 다르게 동작해야 할 때 필요
        )

        x = self.layer_norm_2(x + attn_output)  # 어텐션 연산의 출력(attn_output)과 이전 단계의 출력 x를 더한 후, 두 번째 레이어 정규화를 적용
        return x                                # '잔차 연결(residual connection)'과 '레이어 정규화'의 조합으로, 모델의 깊이가 깊어져도 안정적인 학습

class Embeddings(tf.keras.layers.Layer):    # Transformer 모델에서 임베딩 층은 입력 텍스트의 각 단어를 고정된 크기의 벡터로 변환하는 역할을 하며, 단어의 위치 정보를 모델에 제공하는 위치 임베딩도 함께 사용

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(      # embed_dim: 임베딩 벡터의 차원으로, 각 단어를 표현할 벡터의 크기
            vocab_size, embed_dim)                              # max_len: 입력 시퀀스의 최대 길이로, 모델이 처리할 수 있는 입력 시퀀스의 최대 단어 수
        self.position_embeddings = tf.keras.layers.Embedding(   # 위치 임베딩 정의,  max_len : 위치 임베딩을 위한 어휘 사전의 크기 역할
            max_len, embed_dim, input_shape=(None, max_len))    # embed_dim은 위치 임베딩 벡터의 차원을 지정하며, 토큰 임베딩과 동일한 차원을 사용. 이는 토큰 임베딩과 위치 임베딩을 결합할 때 차원이 일치해야 하기 때문
                                                                #  input_shape=(None, max_len) 여기서 None은 배치크기이며, 어떤 크기의 배치도 처리할 수 있음을 의미

    def call(self, input_ids):                                      # input_ids는 모델이 처리할 텍스트를 토큰 ID 시퀀스로 변환한 것입니다. 각 ID는 어휘 사전 내의 특정 단어를 나타냄
        length = tf.shape(input_ids)[-1]                            # length는 입력 시퀀스의 길이를 나타냄
        position_ids = tf.range(start=0, limit=length, delta=1)     # 0부터 length-1까지의 위치 ID를 생성. 이는 시퀀스 내 각 토큰의 위치를 나타냄
        position_ids = tf.expand_dims(position_ids, axis=0)         # position_ids에 배치 차원을 추가합니다. 이는 위치 임베딩을 위해 필요한 형태의 조정

        token_embeddings = self.token_embeddings(input_ids)             # 각 토큰 ID는 고정된 차원의 벡터로 변환
        position_embeddings = self.position_embeddings(position_ids)    # 생성된 위치 ID에 대응하는 위치 임베딩을 조회. 이는 시퀀스 내 각 토큰의 위치 정보를 모델에 제공

        return token_embeddings + position_embeddings                   # 토큰 임베딩과 위치 임베딩을 요소별(element-wise)로 더한다.
                                                                        # 각 토큰의 최종 임베딩은 해당 토큰의 의미적 정보와 위치 정보를 모두 포함

class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):            # units: 피드포워드 신경망(FFN)의 유닛(뉴런) 수.
        super().__init__()
        self.embedding = Embeddings(                        # 입력 토큰 ID를 고정된 크기의 벡터로 변환하는 임베딩 레이어를 생성, MAX_LENGTH는 입력 시퀀스의 최대 길이를 의미
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)     # tokenizer.vocabulary_size()는 토큰화에 사용된 어휘의 크기(어휘 사전의 크기), embed_dim은 임베딩 벡터의 차원

        self.attention_1 = tf.keras.layers.MultiHeadAttention(      
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1         # dropout은 어텐션 스코어를 계산할 때 사용되는 드롭아웃 비율
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")   # 피드포워드 신경망(FFN)을 정의. 첫 번째 Dense 레이어는 units 개의 유닛을 가지며 활성화 함수로 ReLU를 사용 
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)                 # 두 번째 Dense 레이어는 출력 차원을 임베딩 차원으로 맞추기 위해 embed_dim을 사용

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax") # 최종 출력을 위한 Dense 레이어

        self.dropout_1 = tf.keras.layers.Dropout(0.3)                   # 과적합 방지
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    

    def call(self, input_ids, encoder_output, training, mask=None): # input_ids는 입력 토큰의 ID, training은 모델이 훈련 모드인지 예측 모드인지를 나타내는 부울 값?
        embeddings = self.embedding(input_ids)  # 입력 ID를 임베딩 벡터로 변환. 이 임베딩은 시퀀스 내 각 토큰을 고차원 공간에서의 밀집 벡터로 매핑

        combined_mask = None            # combined_mask와 padding_mask를 초기화. 이들은 어텐션 메커니즘에서 특정 위치를 무시하기 위해 사용
        padding_mask = None
        
        if mask is not None:        # mask가 제공되었는지 확인, mask는 일반적으로 입력 시퀀스에서 패딩된 부분, 패딩은 시퀀스의 길이를 동일하게 맞추기 위해 추가된 불필요한 값
            causal_mask = self.get_causal_attention_mask(embeddings)    # 인과적 마스크를 생성, 인과적 마스크 : 모델이 주어진 시점에서 오직 이전의 정보만을 참조하도록 강제함
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)  # 제공된 mask를 3차원 텐서로 확장하고 정수형으로 변환합니다. 이 padding_mask는 시퀀스 내의 패딩 부분을 식별하는 데 사용
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32) # mask를 다시 한 번 변형하여 combined_mask를 생성. 이 과정은 차원을 조정하여 mask가 어텐션 계산에 적합한 형태를 갖추도록 한다.
            combined_mask = tf.minimum(combined_mask, causal_mask)          # combined_mask와 causal_mask를 결합하여 최종 마스크를 생성, 두 마스크 간의 요소별 최소값을 계산
                                                                            # 이는 두 마스크 중 하나라도 어텐션을 차단하는 위치에는 최종 마스크도 어텐션을 차단하도록 보장

        attn_output_1 = self.attention_1(       # 첫 번째 멀티 헤드 어텐션 레이어를 적용
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,       # combined_mask는 특정 위치의 어텐션을 차단하는 데 사용
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)        # 어텐션의 결과와 입력 임베딩을 더한 후 레이어 정규화를 적용
                                                                    # 트랜스포머에서 일반적인 "잔차 연결(residual connection)" 후에 정규화를 수행하는 패턴
        attn_output_2 = self.attention_2(                           # 두 번째 멀티 헤드 어텐션 레이어를 적용
            query=out_1,                                            # 이번에는 디코더의 이전 레이어 출력(out_1)을 쿼리로, 인코더의 출력을 키와 값으로 사용
            value=encoder_output,
            key=encoder_output,                                     # "인코더-디코더 어텐션"이라고 불리며, 디코더가 인코더의 출력을 참조할 수 있게 합니다
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)             # 두 번째 어텐션 레이어의 출력과 이전 레이어의 출력을 더한 후 레이어 정규화를 적용

        ffn_out = self.ffn_layer_1(out_2)                           # 피드포워드 신경망(FFN)을 적용
        ffn_out = self.dropout_1(ffn_out, training=training)        # 첫 번째 Dense 레이어를 통과한 후 드롭아웃을 적용하고, 두 번째 Dense 레이어를 통과
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)                 # FFN의 출력과 이전 레이어의 출력을 더한 후 레이어 정규화를 적용하고,
        ffn_out = self.dropout_2(ffn_out, training=training)        #  드롭아웃을 다시 적용
        preds = self.out(ffn_out)                       # 마지막 Dense 레이어를 통해 최종 예측을 생성, 소프트맥스 활성화 함수를 사용하여 각 토큰의 확률 분포를 출력
        return preds                                                # 계산된 예측을 반환


    def get_causal_attention_mask(self, inputs):            #  인과적(Sequential) 어텐션은 주로 언어 모델링과 같이, 현재 위치에서 이전 위치의 정보만을 참조하여 다음 단어를 예측할 때 사용
        input_shape = tf.shape(inputs)                                  # inputs의 형태를 구하고, 배치 크기(batch_size)와 시퀀스 길이(sequence_length)를 추출
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]                # sequence_length 길이만큼의 범위를 생성하여 i와 j에 할당
        j = tf.range(sequence_length)                               # i에는 새로운 축을 추가하여 형태를 변환 여기서 i와 j는 각각 시퀀스 내 위치 인덱스를 나타냄
        mask = tf.cast(i >= j, dtype="int32")                       # i와 j를 비교하여, i가 j보다 크거나 같을 경우 True를, 그렇지 않을 경우 False를 반환하는 행렬을 생성, 행렬을 정수형(int32)으로 변환
       # 결과적으로, 현재 위치(i)에서 이전 위치(j)를 참조할 수 있도록 하는 마스크가 생성. 현재 위치나 이전 위치에 대한 어텐션은 허용되지만, 미래 위치에 대한 어텐션은 차단
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))    # 마스크의 형태를 (1, sequence_length, sequence_length)로 변환하여, 배치 크기에 관계없이 사용할 수 있도록 함
        mult = tf.concat(                                           # 배치 크기를 포함하는 텐서를 생성하여, 마스크를 해당 배치 크기만큼 복제할 준비를 함
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],      # tf.constant([1, 1], dtype=tf.int32)는 시퀀스 길이와 시퀀스 길이 차원을 유지하기 위한 값
            axis=0
        )
        return tf.tile(mask, mult)       # 마스크를 mult 변수에 정의된 대로 복제
        #  결과적으로, 생성된 마스크는 전체 배치에 대해 적용될 수 있으며, 각 시퀀스에서 현재 위치나 이전 위치로만 어텐션을 수행할 수 있도록 합

def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top = False,
        weights = 'imagenet'
    )

    output = inception_v3.output            # 모델의 출력을 가져옴, 모델의 최상위 층이 제거된 상태라, 이미지의 고차원적인 특징만을 담고있는 텐서가 된다
    output = tf.keras.layers.Reshape(       # 모델의 출력 텐서 형태를 2차원으로 바꿔준다. 모든높이와 너비에 걸쳐있는 특징맵을 일렬로 펼치는 작업이다.
        (-1, output.shape[-1]))(output)     # output.shape[-1]은 채널 차원을 유지
    cnn_model = tf.keras.models.Model(inception_v3.input, output)       # 변형된 출력을 사용하여 새로운 모델을 정의. 이 모델은 InceptionV3의 입력을 받아서, 변형된 출력을 내보내는 구조를 가진다.
    return cnn_model



class ImageCaptioningModel():
    
    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model                      #  클래스의 다른 메소드들에서 이들 컴포넌트에 접근할 수 있게 해줌
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')          # fit과정에서 평균 loss계산
        self.acc_tracker = tf.keras.metrics.Mean(name='accuracy')       # fit과정에서 평균 acc 계산

    def calculate_loss(self, y_true, y_pred, mask):     # 이 메소드는 실제 레이블(y_true), 모델의 예측값(y_pred), 그리고 손실 계산 시 사용할 마스크(mask)를 인자로 받는다.
                                                        # 마스크는 일반적으로 시퀀스 데이터에서 특정 부분을 무시하기 위해 사용
        loss = self.loss(y_true, y_pred)                # 실제 레이블(y_true)과 예측 레이블(y_pred) 사이의 손실을 계산, 계산된 손실 값을 loss 변수에 저장
        mask = tf.cast(mask, dtype=loss.dtype)          # mask를 loss와 같은 데이터 타입으로 형변환, tf.cast 함수는 변수의 데이터 타입을 변경하는 데 사용
                                                        #  마스크의 데이터 타입을 손실 값의 데이터 타입으로 변경하여 후속 연산에서 데이터 타입 불일치 문제를 방지
        loss *= mask        # 손실 값에 마스크를 요소별(element-wise)로 곱함, 마스크의 값은 0또는 1의 값을 가지는데 마스크가 0인 위치에서의 손실은 제외된다. 
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)    # 마스크를 적용한 손실의 합을 마스크 값의 합으로 나누어 평균 손실을 계산
    
    def calculate_accuracy(self, y_true, y_pred, mask): # 이 메소드는 실제 레이블(y_true), 모델의 예측값(y_pred), 그리고 손실 계산 시 사용할 마스크(mask)를 인자로 받는다
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))  # 예측된 레이블(y_pred)의 최대 값(즉, 가장 높은 확률을 가진 클래스의 인덱스)이 실제 레이블(y_true)과 같은지 비교
                            # 다중 클래스 분류 문제에서 클래스를 예측하는 표준 방식. 이 비교 결과는 불리언(Boolean) 텐서로 반환
        accuracy = tf.math.logical_and(mask, accuracy)  # 마스크와 앞서 계산된 정확도 간의 논리적 AND 연산을 수행,마스크가 True(또는 1)인 위치에서만 정확도 값을 유지하고
                                                        # 그렇지 않은 경우(즉, 패딩된 위치)에는 False로 설정, 결국 패딩된 부분을 정확도 계산에서 제외
        accuracy = tf.cast(accuracy, dtype=tf.float32)  # 불리언 텐서인 accuracy를 실수형(float32)으로 변환합니다. 이는 정확도를 계산하기 위해 필요한 수치 연산을 가능하게 합니다
        mask = tf.cast(mask, dtype=tf.float32)          #  마스크를 실수형(float32)으로 변환, 마스크와 정확도 값을 동일한 데이터 타입으로 맞추고, 이후의 수치 연산을 용이하게 함
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)    # 마스크가 적용된 정확도의 총합을 마스크가 적용된 요소의 총 수로 나누어 평균 정확도를 계산
                                                                # 스크된(패딩된) 부분을 제외하고 실제 데이터에 대한 모델의 정확도를 평가

    def compute_loss_and_acc(self, img_embed, captions, training=True): # 이미지 임베딩, 캡션, 그리고 훈련 중인지 여부를 나타내는 training 플래그를 인수로 받음
        encoder_output = self.encoder(img_embed, training = True)       # encoder는 이미지 임베딩을 입력으로 받아서 인코딩된 특성을 출력
        y_input = captions[:, :-1]      # 각 캡션의 마지막 토큰을 제외한 부분을 y_input으로 사용, 디코더에 입력으로 주어지며, 각 캡션의 시작부터 끝까지를 예측하는데 사용
        y_true = captions[:, 1:]        # y_true는 각 캡션의 첫 토큰을 제외한 모든 토큰을 실제 값으로 사용, 모델의 예측을 실제 값과 비교하는 데 사용
        mask = (y_true != 0)        # mask는 y_true에서 0이 아닌 값을 가지는 위치를 나타냄, 손실과 정확도 계산 시 패딩된 부분을 무시하도록 함
        y_pred = self.decoder(      # decoder는 y_input, 인코더의 출력(encoder_output), 그리고 mask를 입력으로 받아 예측된 캡션을 출력
            y_input, encoder_output, training=True, mask=mask)  
        loss = self.calculate_loss(y_true, y_pred, mask)    # calculate_loss 메서드는 y_true, y_pred, 그리고 패딩 마스크를 사용하여 손실을 계산
        acc = self.calculate_accuracy(y_true, y_pred, mask) # calculate_loss 메서드는 y_true, y_pred, 그리고 패딩 마스크를 사용하여 정확도를 계산
        return loss, acc            # 계산된 loss와 acc를 반환
    
    def train_step(self, batch):        # 하나의 배치에 대한 훈련 단계를 정의,  batch는 이미지와 해당 캡션을 포함
        imgs, captions = batch          # 입력된 batch에서 이미지(imgs)와 캡션(captions)을 추출
        
        if self.image_aug:              # 만약 이미지 증강(image_aug)이 활성화되어 있다면
            imgs = self.image_aug(imgs)     # 이미지 증강을 이미지 배치에 적용
            
            img_embed = self.cnn_model(imgs)        # cnn_model을 사용하여 이미지 배치를 임베딩으로 변환, CNN 모델은 이미지의 특징을 추출하는 역할
            
            with tf.GradientTape() as tape:  # GradientTape를 사용하여 자동 미분을 위한 컨텍스트를 생성, 학습 가능한 변수에 대한 손실 함수의 기울기를 계산하는 데 사용
                loss, acc = self.compute_loss_and_acc(      # 현재 배치에 대한 손실과 정확도를 계산
                    img_embed, captions
                )
            train_vars = (                  # 훈련 가능한 변수들을, 인코더와 디코더의 훈련 가능한 변수들의 합으로 정의
                self.encoder.trainable_variables + self.decoder.trainable_variables #  이 변수들에 대한 기울기를 계산하여 모델을 업데이트할 때 사용
            )
            grads = tape.gradient(loss, train_vars)         # tape.gradient를 사용하여 주어진 손실에 대한 훈련 가능한 변수들의 기울기를 계산
            self.optimizer.apply_gradients(zip(grads, train_vars))  # 계산된 기울기와 변수들을 옵티마이저에 적용하여 모델의 가중치를 업데이트
            self.loss_tracker.update_state(loss)            # 손실과 정확도 추적기를 업데이트하여 현재 훈련 단계의 손실과 정확도를 기록
            self.acc_tracker.update_state(acc)
            
            return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}  # 최종적으로, 현재까지의 평균 손실과 정확도를 반환

    def test_step(self, batch):         #  테스트 배치를 처리, 모델이 새로운 데이터에 대해 어떻게 수행하는지 평가
        imgs, captions = batch          # 입력된 batch에서 이미지(imgs)와 해당 캡션(captions)을 분리
        
        img_embed = self.cnn_model(imgs)        # self.cnn_model을 사용하여 입력 이미지를 임베딩으로 변환, 이미지에서 중요한 특성을 추출하는 데 사용
        
        loss, acc = self.compute_loss_and_acc(      #  training=False는 모델이 평가 모드에 있음을 나타냄, 드롭아웃과 같은 훈련 중에만 적용되는 기술들이 비활성화
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)        # 현재 배치의 손실과 정확도를 추적. 이는 전체 테스트 데이터셋에 대한 평균 손실과 정확도를 계산하는데 사용
        self.acc_tracker.update_state(acc)
        
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    
    @property
    def metrics(self):          # metrics 프로퍼티는 모델의 평가 중에 사용되는 메트릭을 반환
        return [self.loss_tracker, self.acc_tracker]        # 훈련이나 테스트 과정에서 모델의 성능을 추적하는데 사용
        

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)         # 인코더 층을 생성, 임베딩 벡터의 차원을 지정, 두 번째 인자 1은 인코더 레이어 내의 헤드(head)의 수를 의미
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)  # UNITS는 디코더의 내부 유닛(또는 차원) 수


cnn_model = CNN_Encoder()           #  이미지 캡셔닝 모델을 구성하는 주요 요소들을 초기화하고, 이를 바탕으로 전체 모델을 구축하는 과정
caption_model = ImageCaptioningModel(   # 각 요소는 이미지로부터 의미 있는 특징을 추출하고, 이를 사용하여 이미지에 대한 설명을 생성하는 데 핵심적인 역할을 수행
    cnn_model = cnn_model, encoder=encoder, decoder=decoder, imag_aug = image_augmentation)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(  #  함수에 입력되는 y_pred이 로짓(즉 소프트맥스 함수 적용 전의 값) 형태인지, 확률 형태인지를 지정
    from_logits=False, reduction="none" # 예측값이 이미 확률로 변환된 상태(즉, 모델의 출력층에 소프트맥스 활성화 함수가 적용되었음)
)   # reduction: 손실 값들을 어떻게 집계할 것인지를 결정. default는 "auto" : 배치 내의 손실을 평균내는 것을 의미
    # "none"으로 설정하면, 각 샘플에 대한 손실이 개별적으로 계산되어 배치의 각 요소에 대한 손실을 그대로 반환
    
early_stopping = tf.keras.callbaacks.EarlyStopping(patience=5, restore_best_weights=True, monitor = 'val_acc')

caption_model.compile(
    optimizer= tf.keras.optimizers.Adam(),
    loss=cross_entropy
)

history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)         # 주어진 경로의 이미지 파일을 읽고, 그 내용을 바이트 문자열로 반환
    img = tf.io.decode_jpeg(img, channels=3)    #  읽어들인 바이트 문자열(img)을 디코딩하여 RGB 채널을 가진 이미지 텐서로 변환, RGB
    img = tf.keras.layers.Resizing(299, 299)(img)   # Inception V3 모델은 299x299 크기의 이미지를 입력으로 받도록 설계되었음
    img = tf.keras.applications.inception_v3.preprocess_input(img)  #  이미지의 픽셀 값 범위를 조정, Inception V3의 경우, 입력 픽셀 값의 범위를 [-1, 1]로 조정
    return img              #  예를 들어, 원본 이미지 픽셀 값이 [0, 255] 범위에 있을 경우 이를 모델이 기대하는 입력 범위로 변환 



def generate_caption(img_path, add_noise=False):    # add_noise=True :이미지에 임의의 잡음을 추가,  잡음이 추가된 후, 이미지는 0에서 1 사이의 값으로 정규화. 하지만 우린 잡음 안넣음
    img = load_image_from_path(img_path)            # 주어진 경로(img_path)에서 이미지를 로드

    if add_noise:                                   # add_noise가 True일 경우, 이미지에 노이즈를 추가하는 코드 블록
        noise = tf.random.normal(img.shape)*0.1     #  원본 이미지와 같은 형태의 노이즈를 생성하고, 이를 원본 이미지에 추가
        img = img + noise                           # , 이미지를 0과 1 사이의 값으로 정규화하여 노이즈 추가 후에도 이미지의 픽셀 값이 유효한 범위 내에 있도록 함
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    img = tf.expand_dims(img, axis=0)               # 이미지 텐서에 배치 차원을 추가, axis=0은 배치 차원을 가장 앞에 추가
    img_embed = caption_model.cnn_model(img)        # 이미지 캡션 모델의 CNN 부분을 사용하여 입력 이미지를 임베딩(특징 벡터)으로 변환, 이 임베딩은 이미지의 중요한 특징을 추출한 벡터로, 캡션 생성에 사용
    img_encoded = caption_model.encoder(img_embed, training=False)  # 임베딩된 이미지를 인코더에 통과시켜 추가적으로 인코딩

    y_inp = '[start]'                               # 캡션 생성을 시작하기 위한 초기 토큰 [start]를 설정
    for i in range(MAX_LENGTH-1):                   # 최대 캡션 길이까지 반복하면서 각 시점에서 가장 가능성 있는 단어를 예측, -1은 캡션의 마지막에 [end] 토큰을 추가하기 위한 공간을 남겨둠
        tokenized = tokenizer([y_inp])[:, :-1]      # 현재까지의 캡션(y_inp)을 토큰화하고, 마지막 토큰을 제외한 모든 토큰을 사용하여 다음 단어의 예측에 사용, 이는 다음 단어를 예측하기 위한 입력으로 사용
        mask = tf.cast(tokenized != 0, tf.int32)    # 토큰화된 입력에 대한 마스크를 생성, 실제 단어가 있는 위치는 True로, 패딩 부분은 False로 표시 즉, 패딩된 부분 무시
        pred = caption_model.decoder(               # 디코더를 사용하여 다음 단어를 예측 
            tokenized, img_encoded, training=False, mask=mask)  # img_encoded는 이미지의 인코딩된 특징이며, mask는 앞서 생성한 마스크
        
        pred_idx = np.argmax(pred[0, i, :])             # 가장 높은 확률을 가진 단어의 인덱스를 선택
        pred_idx = tf.convert_to_tensor(pred_idx)       # 선택된 인덱스를 텐서로 변환
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')  # 인덱스를 단어로 변환, numpy().decode('utf-8')는 텐서를 문자열로 변환
        if pred_word == '[end]':                        # 만약 예측된 단어가 [end] 토큰이면, 캡션 생성을 종료
            break
        
        y_inp += ' ' + pred_word                        # 예측된 단어를 현재까지의 캡션에 추가
    
    y_inp = y_inp.replace('[start] ', '')               # 최종 생성된 캡션에서 시작 토큰을 제거
    return y_inp




idx = random.randrange(0, len(captions))    # 0부터 captions 데이터프레임의 길이(행의 개수) 사이에서 무작위로 하나의 정수 인덱스를 생성
img_path = captions.iloc[idx].image         # 데이터프레임에서 무작위로 선택된 이미지의 경로를 img_path에 할당하는 과정


pred_caption = generate_caption(img_path)
print('Predicted Caption:', pred_caption)
print()
Image.open(img_path)

img_url = "https://images.squarespace-cdn.com/content/v1/5e0e65adcd39ed279a0402fd/1627422658456-7QKPXTNQ34W2OMBTESCJ/1.jpg?format=2500w"

im = Image.open(requests.get(img_url, stream=True).raw)     #  웹에서 이미지를 스트리밍 방식으로 다운로드
im = im.convert('RGB')
im.save('tmp.jpg')

pred_caption = generate_caption('tmp.jpg', add_noise=False)
print('Predicted Caption:', pred_caption)
print()
im.show()
