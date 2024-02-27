import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
# import collections
import random
# import requests
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
        super().__init()                                # super().__init__()를 호출하여 부모 클래스(tf.keras.layers.Layer)의 생성자를 초기화합니다.
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

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds


    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)





encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)         # 인코더 층을 생성, 임베딩 벡터의 차원을 지정
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)



























    












