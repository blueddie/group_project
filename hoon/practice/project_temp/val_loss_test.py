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
BASE_PATH = 'd:/_data/coco/archive/coco2017'

with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f: # 경로 안에 json 파일을 읽기모드로 열고 핸들러를 변수 f 에 할당
    data = json.load(f)                # 핸들러 f 를 통해 josn파일을 읽음. 파이썬 딕셔너리형태로 data에 저장됨
    data = data['annotations']         # 데이터에 'annotations'키에 해당하는 데이터만 추출해 data에 다시 덮어쓰기. caption데이터만을 담고있음

img_cap_pairs = []  # 리스트 생성

for sample in data:
    img_name = '%012d.jpg' % sample['image_id'] # image_id를 12자리 숫자로 변환
    img_cap_pairs.append([img_name, sample['caption']]) # img_cap_paris 에 이미지 이름이랑 캡션 쌍으로 저장

captions = pd.DataFrame(img_cap_pairs, columns = ['image', 'caption'])  # img_cap_pairs로 데이터프레임 생성
captions['image'] = captions['image'].apply(
    lambda x: f'{BASE_PATH}/train2017/{x}'      
)       # 데이터 프레임에 있는 이미지 파일의 경로 수정. == image_id를 12자리 숫자로 전부 바꾸는거
captions = captions.sample(70000)   # 샘플 7만개만 사용
captions = captions.reset_index(drop=True)  # drop=True로 기존 captions에 있던거 날리고 7만개만 담기게 함
# print(captions.head())                    # 샘플링 후 인덱스가 0부터 다시 시작되는 임의의 7만개 데이터샘플의 데이터프레임이 됨

def preprocess(text):
    text = text.lower()     # 모두 소문자로
    text = re.sub(r'[^\w\s]', '', text)     # 단어와 공백을 제외한 모든 문자 제거.
    text = re.sub('\s+', ' ', text)     # 공백이 여러개면 하나로 줄임 / 문자 제거하면서 생긴 추가 공백 제거
    text = text.strip()                 # 텍스트 양쪽 끝에 공백 제거
    text = '[start] ' + text + ' [end]'     # 텍스트 양끝에 [start] 와 [end] 추가. 모델이 시작과 끝을 인식하게끔 하기 위함
    return text

captions['caption'] = captions['caption'].apply(preprocess) # 실제 데이터에 전처리 함수를 적용하고 나온값을 다시 저장
# print(captions.head())

# random_row = captions.sample(1).iloc[0]   # 데이터 프레임 내의 무작위 1개의 행 선택, 그 중 첫번째 행 = img
# print(random_row.caption)  # 이게 왜 됨? / 위에서 나온 사진에 해당하는 caption 생성
# print()   # 터미널에서 빈공간 보려고 걍 넣음
# im = Image.open(random_row.image) / 해당하는 caption과 직접 대조하기 위해 이미지 출력
# im.show()

MAX_LENGTH = 40
VOCABULARY_SIZE = 20000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 30

tokenizer = tf.keras.layers.TextVectorization(  # text를 input으로 넣기 위한 text-vectorization
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH
)
tokenizer.adapt(captions['caption'])    # caption열에 있는 텍스트 데이터를 textvectorization에 적용. 단어장 생성
# print(tokenizer.vocabulary_size())  # 단어 대충 12000개 좀 안됨. 누를때마다 바뀜 아마 random_row 때문인거같음

pickle.dump(tokenizer.get_vocabulary(), open( BASE_PATH + '/vocab_coco.file', 'wb'))  # vocab_coco.file 을 열어서 현재 단어장 새로 저장

word2idx = tf.keras.layers.StringLookup(    # 주어진 단어를 해당 단어장의 정수 인덱스로 변환시킴. <- 입력으로 사용하기 위함
    mask_token="",                          # 빈 마스크 토큰 생성
    vocabulary=tokenizer.get_vocabulary())  # 단어장 설정. text-vectorization 레이어에서 가져온 단어장 사용

idx2word = tf.keras.layers.StringLookup(    # word2idx에서 생성된 인덱스를 다시 단어로 바꿔주는 코드
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

img_to_cap_vector = collections.defaultdict(list)   # 빈 리스트가 기본값인 딕셔너리 생성. 새로운 키에 대해서 에러 대신 빈 리스트를 생성해줌
for img, cap in zip(captions['image'], captions['caption']):  # zip함수로 병렬 반복하면서 image -> img, caption -> cap 으로 바꿔서 가져옴
    img_to_cap_vector[img].append(cap)  # 새로 반복중인 img를 키로 사용해 img_to_cap_vector 딕셔너리에 cap을 추가함. 
                                        # 여기서 이미지가 딕셔너리에 없다면 빈 리스트가 자동 생성(86번줄), 이 리스트에 캡션을 추가함

img_keys = list(img_to_cap_vector.keys())   # img_to_cap_vector의 키를 불러옴 == img 파일 이름 리스트
random.shuffle(img_keys)                    # 이미지 셔플. train_test_split 에 있는 shuffle과 같은 역할

slice_index = int(len(img_keys)*0.8)        # 이미지 키 리스트에 80% 지점에 해당하는 인덱스 계산.
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index],   # train_keys에 처음부터 80% 지점 직전 까지
                                          img_keys[slice_index:])   # val_keys에 80% 지점 부터 끝까지.

train_imgs = []         # 빈 리스트 생성
train_captions = []     # 빈 리스트 생성    
for imgt in img_name_train_keys:    # coco 데이터셋은 이미지당 5개의 캡션이 있기때문에 이미지1:캡션1, 이미지1:캡션2 ... 이미지1:캡션5 이렇게 총 다섯장 저장시킴.
    capt_len = len(img_to_cap_vector[imgt]) # 현재 이미지에 대한 캡션 갯수를 구함. 캡션의 리스트 길이를 나타냄
    train_imgs.extend([imgt] * capt_len)    # 각 이미지에 해당하는 캡션들의 개수만큼 해당 이미지 파일을 train_imgs 리스트에 추가
    train_captions.extend(img_to_cap_vector[imgt])  # img_to_cap_vector 딕셔너리에 현재 이미지 파일에 해당하는 캡션을 모두 추가, train_captions에 저장
print(train_captions[0:5])

val_imgs = []       # 빈 리스트 생성
val_captions = []   # 빈 리스트 생성
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv]) # 현재 이미지에 대한 캡션의 갯수를 구함. 캡션 리스트의 길이를 나타냄
    val_imgs.extend([imgv] * capv_len)  # 각 이미지파일에 대한 캡션 개수만큼 해당 이미지의 파일을 리스트에 반복해서 추가, val_imgs라는 list에 저장
    val_captions.extend(img_to_cap_vector[imgv])    # img_to_cap_vector 딕셔너리에 현재 이미지 파일에 해당하는 캡션을 모두 추가, val_captions에 저장
print(len(train_imgs), len(train_captions), len(val_imgs), len(val_captions))   # 55994 55994 14006 14006
# print(train_imgs[:1], train_captions[:1])

def load_data(img_path, caption):   # 함수 정의
    img = tf.io.read_file(img_path) # img_path에 있는 이미지 불러오기
    img = tf.io.decode_jpeg(img, channels=3)    # jpeg를 디코딩, rgb이미지로 변환
    img = tf.keras.layers.Resizing(299, 299)(img)   # 이미지 사이즈를 (299,299)로 바꿈
    img = tf.keras.applications.inception_v3.preprocess_input(img)  # inception_v3가 요구하는 input 형식으로 데이터 전처리.
    caption = tokenizer(caption)    # 캡션 데이터를 토큰화. 토크나이저 적용.
    return img, caption             # 전처리된 이미지와 토큰화된 캡션을 반환함  

train_dataset = tf.data.Dataset.from_tensor_slices( # 리스트나 배열같은 텐서로부터 데이터셋 생성
    (train_imgs, train_captions))               # train_img는 이미지파일 경로 리스트, train_captions는이미지에 해당하는 캡션의 리스트

train_dataset = train_dataset.map(                  
    load_data, num_parallel_calls=tf.data.AUTOTUNE  # load_data(코드 115번줄)로 코드 123번줄 train_dataset 전처리, numparallel_calls 는 로드와 전처리를 병렬로 처리. 성능을  최적화시켜줌
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)        # .shuffle(BUFFER_SIZE)는 전체 데이터를 BUFFER_SIZE 만큼 덩어리씩 묶은뒤 그 덩어리들을 섞는것. 덩어리 내부는 섞이지 않음
                                                        # model.fit의 batchsize 가 여기 붙은것
val_dataset = tf.data.Dataset.from_tensor_slices(   # 리스트나 배열같은 텐서로부터 데이터셋을 생성함
    (val_imgs, val_captions))                       # val_imgs 는이미지파일들의 결로를 담고 있는 리스트, val_captions는 각 이미지에 해당하는 캡션들을 담고있는 리스트

val_dataset = val_dataset.map(                      # 코드 126번줄 train_dataset과 똑같음 이름만 다름
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

image_augmentation = tf.keras.Sequential(           # 데이터 증폭. Sequential 정의
    [   tf.keras.layers.RandomFlip("horizontal"),   # 수평방향으로 무작위로 뒤집음
        tf.keras.layers.RandomRotation(0.2),        # 최대 20% 각도로 무작위 회전 == 최대 72도
        tf.keras.layers.RandomContrast(0.3),        # 픽셀에 무작위한 대비 변화를 적용. 밝기와 명암이 조절됨
    ]
)
def CNN_Encoder():  # InceptionV3를 이용한 CNN_Encoder 함수 정의 // 이미지의 특성을 추출하는 부분
    inception_v3 = tf.keras.applications.InceptionV3(   # keras 제공 InceptionV3. 이미지 분류를 위한 CNN 아키텍처
        include_top=False,                              # 원래 InceptionV3는 이미지 분류기 인데, 이 분류기의 최상층 레이어는 이미지 분류에만 쓰이기 때문에 불러오지 않고 특성 추출 부분까지만 불러오는 구조
        weights='imagenet'                              # ImageNet 데이터셋으로 학습된 가중치 사용. 
    )

    output = inception_v3.output                    # inception_v3를 거쳐서 나온 아웃풋을 ouput에 저장
    output = tf.keras.layers.Reshape(               # (높이,넓이,채널)형식의 3차원 형태에서 2차원 벡터로 차원 축소
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)   # 새 모델을 만들어서 인풋은 inception_v3의 인풋을, 출력은 inception_v3를 거친 출력을사용
    return cnn_model        # cnn_model은 이미지를 입력으로 받아서 InceptionV3로 이미지 특성을 추출하는 모델

class TransformerEncoderLayer(tf.keras.layers.Layer):   # 트랜스포머의 인코더레이어

    def __init__(self, embed_dim, num_heads):       # embed_dim = 임베딩 차원, num_heads는 어텐션 헤드의 갯수
        super().__init__()              # __init__메서트에서 클래스 초기화
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()    # 입력을 정규화하는 레이어 생성. 트랜스포머에서는 레이어 정규화가 중요한 역할을함
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()    # 
        self.attention = tf.keras.layers.MultiHeadAttention(        # 다중헤드어텐션 연산 수행
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu") # 입력을 임베팅차원에 대비해서 확장시키기 위해 Dense레이어를 하나 추가함. embed_dim 만큼 노드(=차원)이 늘어남
    

    def call(self, x, training):    # call에서 실제로 입력데이터를 처리함. x가 데이터, training은 현재 학습중인지 묻는것
        x = self.layer_norm_1(x)    # 입력 데이터를 첫번째 레이어 정규화에 통과시킴
        x = self.dense(x)           # 처리된 입력을 임베딩 차원에 대비해서 확장시키는 Dense레이어에 통과시킴

        attn_output = self.attention(   # 멀티헤드 어텐션 연산 수행. 어텐션 출력 attn_ouput에 저장
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training       # 훈련인지 테스트인지 나타내고 dropout같은 정규화 기법이 적용될지를 결정함. test모드라면 적용안됨
        )

        x = self.layer_norm_2(x + attn_output)  # 입력과 어텐션출력을 더하고 두번째 레이어 정규화 실행
        return x            # residual connection이라고도 하며 입력과 출력간의 차이를 줄여줌. 입력과 어텐션 출력을결합한 후 정보를 안정화시키고 학습을 도와줌

class Embeddings(tf.keras.layers.Layer):        # 트랜스포머 모델의 입력으로 사용될 토큰 및 위치 임베딩을 생성
            # 임베딩이란 : 이미지의 저차원적 특성 벡터를 추출해 유사도가 높은 단어끼리는 임베딩 공간상에서 서로 가까운 곳에 위치하게 됨. 즉 유사성을 띈 단어들 간의 분류를 위함.
    def __init__(self, vocab_size, embed_dim, max_len): 
        super().__init__()      # tf.keras.layers.Layer 를 상속받은 Embeddings 클래스이기 때문에 tf.keras.layers.Layer의 생성자를 호출. 부모 클래스의 모든 속성과 메서드를 상속받게됨
        self.token_embeddings = tf.keras.layers.Embedding(  # 단어 집합의 크기(vocab_size)와 임베딩 차원(embed_dim)을 인자로 받아 각 단어를 고정된 길이의 밀집 벡터로 임베딩
            vocab_size, embed_dim)              # 단어의 의미보존, 차원축소, 단어간 관계표현 
        self.position_embeddings = tf.keras.layers.Embedding(   # 위치 임베딩. 입력 시퀀스의 위치정보를 표현하기위해 위치 임베딩을 사용.
            max_len, embed_dim, input_shape=(None, max_len))    # 입력 시퀀스 안에서 각 토큰의 상대적인 위치를 나타내는임베딩임
    

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]    # input_ids 텐서의 마지막 차원의 길이를 구함. 입력 시퀀스의 길이를 나타냄
        position_ids = tf.range(start=0, limit=length, delta=1) # length만큼의 길이를 가지고 0부터 1씩 증가하는 숫자배열 생성. / 입력 시퀀스의 위치정보를 나타내는 벡터
        position_ids = tf.expand_dims(position_ids, axis=0) # position_ids텐서에 차원을 추가해 (1,length)로 변경함 / 임베딩에 적절한 벡터형태로 변환

        token_embeddings = self.token_embeddings(input_ids) # 입력으로 받은 토큰 시퀀스에 대한 토큰 임베딩 계산. 입력 시퀀스(문장) 내 각 토큰에 대한 단어 임베딩임.
        position_embeddings = self.position_embeddings(position_ids)    # 입력 시퀀스내 각 토큰의 순서를 나타내는 임베딩. 각 토큰(단어)의 위치 정보에 대한 임베딩임.

        return token_embeddings + position_embeddings   # 토큰 임베딩과 위치 임베딩을더해 최종 임베딩 생성 및 반환. 
    # 이제 입력 시퀀스의 각 토큰에 대한 토큰임베딩과 위치임베딩을 결합한 결과를 얻을수 있음 / 여기서 더하는건 실제 덧셈 연산을 진행하는것
class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()      # tf.keras.layers.Layer클래스 생성자 정의. 여러 구성층 초기화.
        self.embedding = Embeddings(        # 클래스의 객체를 생성해 디코더 레이어의 임베딩 층을 초기화
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH) # 각각 토크나이저에 담긴 단어의 사이즈, 임베딩 차원, 최대 시퀀스의 길이를 나타냄

        self.attention_1 = tf.keras.layers.MultiHeadAttention(  # 첫번째 멀티헤드 어텐션 층을 초기화함. 
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1 # 각각 어텐션 헤드의 수, 어텐션 키의 차원, 드롭아웃설정을 의미함
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(  # attention_1과 같음
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization() # 첫번째 레이어 정규화층을 초기화함.
        self.layernorm_2 = tf.keras.layers.LayerNormalization() # 레이어 정규화는 네트워크의 출력을 표준 정규 분포로 정규화해서 학습을 안정화 시키는 기법
        self.layernorm_3 = tf.keras.layers.LayerNormalization() # Gradient vanishing 방지, 과적합 방지

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")  # Feedforward Neural Network 레이어 정의. units 개수의 노드로 뻗어감
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim) # embed_dim 개수의 노드를 가진 완전연결 레이어
        # ffn 레이어는 트랜스포머 디코더의 각 서브층에서 사용됨
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax") # 출력 레이어. 토크나이저 어휘의 크기만큼의 뉴런(노드)를 가짐.
        # softmax를 이용해 다음 단어를 예측하는데 사용됨
        self.dropout_1 = tf.keras.layers.Dropout(0.3)   # 드롭아웃 0.3
        self.dropout_2 = tf.keras.layers.Dropout(0.5)   # 드롭아웃 0.5
        # 디코더 계층의 출력 레이어 및 드롭아웃 레이어 정의

    def call(self, input_ids, encoder_output, training, mask=None): # inpud_ids: 입력 토큰 시퀀스, encoder_output: 인코더의 출력, training: 학습 여부를 묻는 boolian
        embeddings = self.embedding(input_ids)                      # mask: 어텐션 매커니즘의 마스크(특정 위치 무시)를 쓸건지. / 연관이 떨어지는 단어에 마스크를 씌워서 무시할건지

        combined_mask = None    # 디코더의 셀프 어텐션을 위한 마스크
        padding_mask = None     # 패딩 토큰을 처리하기위한 마스크
                                # 이 두 마스크를 결합해 디코더 계층에서 어텐션을 수행할 때 사용됨
        if mask is not None:    # 마스크가 None 이 아니라면 실행될 코드
            causal_mask = self.get_causal_attention_mask(embeddings)    # 현재 위치에서 고정. 이후의 정보가 현재 위치에 영향을 주지 않도록 하는 코드
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)  # 시퀀스의 패딩된 부분을 나타내며, 패딩 토큰이 있는 위치를 확인해 모델이 패딩 토큰을 무시하도록함.
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32) # 미래의 위치와 패딩된 위치 모두에 대한 마스크를 포함하고 이 위치정보를 무시하도록함
            combined_mask = tf.minimum(combined_mask, causal_mask)  # 위의 두 마스크를 결합. // 새로운 축을 생성해 위치를 단일 차원에 유지시켜서 이동하지 않도록하는것.

        attn_output_1 = self.attention_1(   
            query=embeddings,   # query, value, key값을 지정하고 임베딩된 데이터인 embeddings를 사용
            value=embeddings,   # query란 어텐션 메커니즘에서 주목해야할 대상을 나타냄. 어텐션을 계산할 때 기준이 되는 값
            key=embeddings,
            attention_mask=combined_mask,   # 어텐션 마스크는 어텐션 연산시 특정 위치의 단어를 가림. 위에서 정의한combined_mask 사용
            training=training   # train인지 test인지 여부를 나타냄
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)    # 첫번째 어텐션레이어의 아웃풋과 입력 임베딩을 더하고 정규화를 적용. (코드 216번줄에서 정의된 레이어 사용)

        attn_output_2 = self.attention_2(   # 이전 출력 out_1으로 query를 계산, 인코더의 출력으로 어텐션을 계산함
            query=out_1,            # 이번엔 어텐션의 기준을 out_1으로 다시 계산함.
            value=encoder_output,   # 코드 156번줄 인코더 레이어의 아웃풋을 key, value로 사용해 어텐션 계산
            key=encoder_output,
            attention_mask=padding_mask,    # 패딩 토큰의 위치 마스크를 사용
            training=training       # 훈련 여부 boolian
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2) # 첫번째 어텐션과 두번째 어텐션을 덧셈 연산 후 정규화 진행 / 코드 217번줄
                                            # 밀집층이란 : 입력 뉴런과 출력 뉴런이 모두 연결된 완전 연결층을 의미. 여러개의 뉴런으로 구성된 레이어를 밀집층이라고 부름
        ffn_out = self.ffn_layer_1(out_2)   # out_2가 코드 220번줄 밀집레이어 통과. relu적용
        ffn_out = self.dropout_1(ffn_out, training=training)    # 그대로 dropout 적용
        ffn_out = self.ffn_layer_2(ffn_out) # dropout 적용 후 코드 221번줄의 밀집레이어 통과

        ffn_out = self.layernorm_3(ffn_out + out_2) # 지금 까지 통과시킨 데이터를 out_2와 덧셈 연산 후 레이어 정규화 진행
        ffn_out = self.dropout_2(ffn_out, training=training) # 0.5 짜리 dropout
        preds = self.out(ffn_out)   # softmax를 가진 output_layer통과 / 코드 223번줄
        return preds


    def get_causal_attention_mask(self, inputs):    # 원긴과 결과가 명확한 어텐션 마스크 생성.
        input_shape = tf.shape(inputs)      # 인풋의 shape
        batch_size, sequence_length = input_shape[0], input_shape[1]    # 배치 크기와 시퀀스 길이 추출
        i = tf.range(sequence_length)[:, tf.newaxis]    # 0부터 시퀀스 길이 까지의 인덱스를 생성. 새로운 축을 추가해 위치 저장
        j = tf.range(sequence_length)       # 0부터 시퀀스 길이 까지의 인덱스를 생성
        mask = tf.cast(i >= j, dtype="int32")   # 인덱스를 사용해 인과적인 어텐션 마스크 생성. 이 마스크는 이전 토큰에만 주의를 기울이도록 마스킹
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))    # 마스크를 원하는 모양으로 재구성함
        mult = tf.concat(   # 배치 크기를 고려해 마스크를 복제할 차원을 생성함
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )                           # mult는 마스크를 다중으로 확장하는데 사용됨 / 마스크를 배치크기와 시퀀스 길이에 따라 확장하는데 사용
        return tf.tile(mask, mult)  # 마스크를 복제해서 최종 인과적인 어텐션 마스크를 생성하고 mask에 반환함. 

class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()  # 부모 클래스의 초기화 메서드 호출
        self.cnn_model = cnn_model  # 이미지 특성 추출을 위한 CNN모델(코드 143 번줄)
        self.encoder = encoder  # 이미지 및 텍스트 특성을 인코딩하는 인코더 (코드156 번줄)
        self.decoder = decoder  # 인코딩된 특성을 이용해 캡션 생성 (코드 202 번줄)
        self.image_aug = image_aug  # 데이터 증강 (코드 137 번줄)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")  # 로스 평균을 추적하는 메트릭스
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")   # 정확도 평균을 추적하는 메트릭스


    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)    # model.evaluate 비슷한거임. loss 구하기
        mask = tf.cast(mask, dtype=loss.dtype)  # 마스크를 loss와 동일한 데이터 타입으로 변환
        loss *= mask                        # loss에 mask를 곱해서 마스크된 위치의 로스만 유지
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)    # 마스크된 위치의 loss평균 계산


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))  # 예측값과 실제값을 비교해서 정확도 계산. accuracy_score 같은거
        accuracy = tf.math.logical_and(mask, accuracy)  # 마스크와 정확도를 논리적(logical)으로 조합해 유효한 위치의 정확도만 유지
        accuracy = tf.cast(accuracy, dtype=tf.float32)  # 정확도를 부동소수점 형태로 변환
        mask = tf.cast(mask, dtype=tf.float32)          # 마스크를 부동소수점 형태로 변환
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)    # 유효한 위치의 정확도의 평균계산
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True) # 트랜스포머 인코더에 증폭된 데이터셋 넣기
        y_input = captions[:, :-1]      # 입력 캡션에서 마지막 토큰을 제외한 부분만 y_input으로 사용 / 디코더 모델이 예측에 사용할 데이터라서 마지막은 안줌
        y_true = captions[:, 1:] # 입력 캡션에서 첫번째 토큰을 제외한 부분만 y_true로 사용 / 디코더 모델이 예측해야할 다음 토큰과 비교를 위해 첫번째 토큰을 제외
        mask = (y_true != 0)            # 패딩된 부분을 제외한 실제 토큰을 나타내는 마스크 생성
        y_pred = self.decoder(          # 트랜스포머 디코더에 입력 캡션과 인코더의 output을 입력으로 사용, y_pred에 저장
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)        # 로스 계산
        acc = self.calculate_accuracy(y_true, y_pred, mask)     # 정확도 계산
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch  # 배치사이즈만큼 이미자와 캡션을 전달

        if self.image_aug:      # self.image_aug가 설정되어있는지 확인
            imgs = self.image_aug(imgs) # 적용 되어있다면 .imgs에 증강데이터 적용
        
        img_embed = self.cnn_model(imgs) # 이미지를 cnn에 통과시켜 이미지 임베딩을 얻음 == 특성값 추출

        with tf.GradientTape() as tape: # 텐서의 자동 미분기능을 위한 코드. 손실함수와 그래디언트 계산 가능
            loss, acc = self.compute_loss_and_acc(  # 이미지 임베딩과 캡션데이터 전달해서 로스와 정확도 계산
                img_embed, captions
            )
    
        train_vars = (      # 학습가능한 변수를 수집. 인코더와 디코더의 학습가능 변수를 모아 리스트로 만듦.
            self.encoder.trainable_variables + self.decoder.trainable_variables 
        )
        grads = tape.gradient(loss, train_vars)     # 로스의 그래디언트 계산. 후속 모델의 가중치 업데이트에 사용됨
        self.optimizer.apply_gradients(zip(grads, train_vars))  # 계산된 그래디언트로 가중치 갱신. 옵티마이저로 그래디언트 디센트를 수행.
        self.loss_tracker.update_state(loss)    # 로스 갱신
        self.acc_tracker.update_state(acc)   # 정확도 갱신

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()} # 로스와 정확도를 담은 딕셔너리를 반환함
    

    def test_step(self, batch): 
        imgs, captions = batch  # 배치사이즈 만큼 이미지와 캡션을 불러옴

        img_embed = self.cnn_model(imgs)    # 이미지를 CNN모델에 통과시켜 이미지의 임베딩값을 얻음. == 특성 추출

        loss, acc = self.compute_loss_and_acc(      # 로스와 정확도 계산. 이미지의 특성값과 캡션데이터가 전달되고 train모드를 꺼서 dropout같은 학습용 레이어 적용 x
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)    # 로스 갱신
        self.acc_tracker.update_state(acc)      # 정확도 갱신

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}   # 딕셔너리로 로스와 정확도 반환

    @property       # metrics를 속성(property)으로 만들어줌. 호출할땐 메서드처럼 호출하지만 실제로는 속성값에 접근하는것처럼 동작하게됨
    def metrics(self):  # 모델의 메트릭스를 반환함. 로스와 정확도가 포함된 리스트를 반환
        return [self.loss_tracker, self.acc_tracker] # 리스트 안에 학습 및 테스트중 로스와 정확도를 추척하는 메트릭스
            # 이 속성(property)는 모델 객체를 통해 쉽게 로스와 정확도를 확인할수 있게함. model.metrics를 호출하면 현재 로스와 정확도를 담은 리스트를 뱉어줌
encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1) # EMBEDDING_DIM의 숫자만큼의 차원을 가진 벡터로 변환됨. ex) 512차원의 임베딩 벡터
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)  # EMBEDDING_DIM의 숫자만큼 차원을 가진 벡터로, UNITS의 숫자만큼 순환 유닛(뉴런)을 가진 8개의 벡터로 생성
                                                            # ex) 512차원의 512개 뉴런을 가진 8개의 벡터
cnn_model = CNN_Encoder()   # 코드 143번줄 CNN인코더
caption_model = ImageCaptioningModel(   # 캡션모델 정의. / CNN인코더, 트랜스포머인코더, 트랜스포머 디코더, 증강된 이미지 사용 하겠다는뜻
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(  # SparseCategoricalCrossentropy 사용
    from_logits=False, reduction="none"         # 입력이확률값인지 로짓값인지 정함. 이번엔 확률이라 False / reduction은 로스의 축소 방법을 지정하는것. "none"이면 각 샘플마다 계산하고 반환함
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor = 'val_acc')

caption_model.compile(  # model.compile
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy  # 코드 374번줄
    # , metrics=['accuracy']
)

history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
# 이 아래는 predict를 위한 함수 설정
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def generate_caption(img_path, add_noise=False):
    img = load_image_from_path(img_path)
    
    if add_noise:
        noise = tf.random.normal(img.shape)*0.1
        img = img + noise
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)
        
        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        
        y_inp += ' ' + pred_word
    
    y_inp = y_inp.replace('[start] ', '')
    return y_inp

idx = random.randrange(0, len(captions))
img_path = captions.iloc[idx].image

pred_caption = generate_caption(img_path)
print('Predicted Caption:', pred_caption)
print()
Image.open(img_path)

img_url = "https://images.squarespace-cdn.com/content/v1/5e0e65adcd39ed279a0402fd/1627422658456-7QKPXTNQ34W2OMBTESCJ/1.jpg?format=2500w"

im = Image.open(requests.get(img_url, stream=True).raw)
im = im.convert('RGB')
im.save('tmp.jpg')

pred_caption = generate_caption('tmp.jpg', add_noise=False)
print('Predicted Caption:', pred_caption)
print()
im.show()

# 가중치 저장
# caption_model.save_weights('c:/Study/project/group_project/min/save/caption_model.h5')
# pickle.dump(caption_model, open('c:/Study/project/group_project/min/caption_model.dat', 'wb'))    # error
# pickle.dump(caption_model, open('c:/Study/project/group_project/min/caption_model.pkl', 'wb'))

# dump(caption_model, 'c:/Study/project/group_project/min/save/caption_model.joblib')