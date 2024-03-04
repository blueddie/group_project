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
import time
# 기본 파일 위치
BASE_PATH = 'd:/_data/coco/archive/coco2017'                                # base 경로 설정

st = time.time()
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
et = time.time()
print(f"걸린 시간 {et - st}")

print(captions.shape)
# captions = captions.sample(10000)
# print(f"추출한 임의의 샘플 : {captions.shape}")      
# # (616767, 2) 정상적으로 데이터프레임 만들었음 확인.
'''
def preprocess(text):
    text = text.lower()                                                     # 텍스트를 소문자로 변환합니다.
    text = re.sub(r'[^\w\s]', '', text)                                     # 특수 문자를 제거합니다.
    text = re.sub('\s+', ' ', text)                                         # 연속된 공백을 하나의 공백으로 대체합니다.
    text = text.strip()                                                     # 문자열 양쪽의 공백을 제거합니다.
    text = '[start] ' + text + ' [end]'                                     # 문장의 시작과 끝을 나타내는 토큰을 추가합니다.
    return text                                                             # 가공한 text 반환

captions['caption'] = captions['caption'].apply(preprocess)                 # captions 데이터프레임의 'caption' 열에 있는 모든 행에 preprocess 함수를 적용합니다.
# print(captions.head())

# random_row = captions.sample(1).iloc[0]                                    # 임의의 하나의 행을 random_row 변수에 할당                                                
# im = Image.open(random_row.image)                                           # random_row 이미지에 접근
# im.show()                                                                  # 이미지 출력
# print(f"출력 이미지 정보 : {random_row}")     
# image      d:/_data/coco/archive/coco2017/train2017/00000...
# caption         [start] a man with a ball of some sort [end]     # 

MAX_LENGTH = 40
VOCABULARY_SIZE = 29630
BATCH_SIZE = 88
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 20

tokenizer = tf.keras.layers.TextVectorization(                              # "토큰(token)"은 텍스트를 작은 단위로 나누는 과정에서의 기본 단위를 의미.
    max_tokens=VOCABULARY_SIZE,                                             # max_tokens: 단어 집합의 크기를 결정. 즉, 텍스트에서 가장 빈도가 높은 상위 n개의 단어만을 사용하여 벡터화
    standardize=None,                                                       # 텍스트 표준화. 텍스트 데이터에 있는 다양한 변형이나 노이즈를 제거하고 일관된 형식으로 만들어주는 과정
    output_sequence_length=MAX_LENGTH                                       # 출력 시퀀스의 길이를 결정합니다. 모든 시퀀스를 동일한 길이로 만들기 위해 필요한 작업.
)

tokenizer.adapt(captions['caption'])
# print(captions['caption'])

print(tokenizer.vocabulary_size())                                          # 나누지 않은 train 전체    29630
# print(f"추출한 임의의 샘플 단어 사전 : {tokenizer.vocabulary_size()}")                                          # 나누지 않은 train 전체    29630


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

train_dataset = train_dataset.map(
    load_data                                                               # 함수를 데이터셋의 각 요소에 적용합니다.
    , num_parallel_calls=tf.data.AUTOTUNE                                   # 데이터 전처리를 병렬로 처리
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)                                # 데이터셋을 셔플링, 이터셋을 배치로 만듭니다.

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


image_augmentation = tf.keras.Sequential(               
    [   tf.keras.layers.RandomFlip("horizontal"),                           # 수평 방향으로 이미지를 무작위로 뒤집습니다.
        tf.keras.layers.RandomRotation(0.2),                                # 이미지를 최대 20도까지 무작위로 회전합니다.
        tf.keras.layers.RandomContrast(0.3),                                # 이미지의 대비를 무작위로 조절합니다.
    ]
)

def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False                                                   # 모델의 상단 부분 (Fully Connected Layer) 을 포함하지 않음. Fully Connected Layer는 모든 입력 뉴런과 출력 뉴런이 서로 연결되어 있는 레이어
                                                                            # 주로 분류나 회귀와 같은 작업에서 사용, 입력 데이터의 특징을 추상화하고 이를 기반으로 예측을 수행, 
                                                                            # 이 레이어를 제거하는 것은 모델의 특징 추출 능력을 강화하고, 특히 이미지 캡션 생성과 같은 작업에서는 시각적 특징을 더 잘 추출
        , weights='imagenet'                                                # # ImageNet 데이터셋으로 사전 훈련된 가중치를 사용합니다.
        
    )
    
    output = inception_v3.output                                            # 모델의 출력을 가져옵니다.
    output = tf.keras.layers.Reshape(                                       # 출력을 재구성하여 3D 텐서를 2D 텐서로 변환
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)           # 모델을 입력과 출력을 지정하여 정의합니다.
    return cnn_model                                                        # 모델 리턴

class TransformerEncoderLayer(tf.keras.layers.Layer):                       # 트랜스포머 모델에서 인코더는 입력 시퀀스의 특성을 추출하고, 입력 시퀀스 내의 각 요소들 간의 상호작용을 모델링하는 역할

    def __init__(self, embed_dim, num_heads):                               # __init__은 파이썬 클래스의 생성자 메서드, embed_dim: 임베딩 차원의 크기, num_heads: 멀티 헤드 어텐션에서 사용되는 어텐션 헤드의 수
        super().__init__()                                                  # 
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()            # LayerNormalization을 사용하여 입력 데이터의 각 차원을 정규화하는 레이어를 초기화
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()            
        self.attention = tf.keras.layers.MultiHeadAttention(                # MultiHeadAttention 레이어를 초기화하여 입력에 대한 멀티 헤드 어텐션 메커니즘을 적용
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")    # Dense 레이어를 초기화하여 입력에 선형 변환을 적용하고 활성화 함수로 ReLU를 사용하여 비선형성을 추가
    

    def call(self, x, training):                                            # 
        x = self.layer_norm_1(x)                                            # 입력 데이터의 각 차원을 정규화합니다.
        x = self.dense(x)                                                   # Dense 레이어를 통해 입력에 선형 변환을 적용합니다.

        attn_output = self.attention(                                       # MultiHeadAttention을 사용하여 셀프 어텐션을 수행합니다.
            query=x,                                                        # 물어보는 주체. 유사한 다른 단어를 찾을 때 사용되는 (질의) 벡터
            value=x,                                                        # 딕셔너리 형태로 저장된다. value는 해당 단어에 대한 구체적 정보를 저장하는 역할
            key=x,                                                          # 쿼리와의 관계를 계산할 단어들. key는 단어의 id 역할?   
                                                                            # "I am a teacher" 라는 문장에서 I 가 다른 단어와 어떤 연관이 있는지 알아보기 위한 self attention을 수행할 때 :
                                                                            # Query : I 에 대한 벡터 Key : am a teacher 각각의 단어 
            attention_mask=None,                                            # 어텐션 마스크를 지정하지 않음을 의미. 어텐션 마스크는 어텐션 계산에 사용되는 가중치를 조절하기 위해 사용됩니다.
            training=training                                               # 훈련 중인지 아닌지를 나타내는 불리언 값입니다. 이것은 모델이 훈련 중인지 추론 중인지에 따라 다르게 작동하는 레이어(예: 드롭아웃)가 있을 때 사용
        )

        x = self.layer_norm_2(x + attn_output)                              # 두 번째 층 정규화를 수행하고, 이전 층의 출력과 어텐션 출력을 더합니다. 이렇게 함으로써, 어텐션에서 얻은 정보를 입력에 누적하고 새로운 특성을 만들어냅니다.
        return x


class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(                  # 토큰 임베딩 레이어 초기화: vocab_size는 어휘 사전의 크기, embed_dim은 임베딩 차원의 크기
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(               # 위치 임베딩 레이어 초기화: max_len은 입력 시퀀스의 최대 길이
            max_len, embed_dim, input_shape=(None, max_len))
    

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]                                    # 입력 시퀀스의 길이 계산
        position_ids = tf.range(start=0, limit=length, delta=1)             # 입력 시퀀스의 각 토큰 위치에 대한 정보 계산 (입력 시퀀스의 각 토큰 위치에 대한 정보를 생성하기 위해 0부터 시작하여 length-1까지의 숫자를 생성)
        position_ids = tf.expand_dims(position_ids, axis=0)                 # position_ids 텐서의 차원을 확장하여 batch 차원을 추가함 position_ids는 입력 토큰의 위치 정보를 나타내는 텐서
                                                                            # 텐서플로우의 멀티-배치(batch) 모델에서는 모든 입력이 배치(batch)의 형태를 갖습니다. 
                                                                            # 즉, 입력 데이터의 첫 번째 차원은 배치 차원입니다. 따라서 입력 토큰의 위치 정보를 나타내는 텐서인 position_ids를 확장하여 배치 차원을 추가해야 합니다.
        
        

        token_embeddings = self.token_embeddings(input_ids)                 # 토큰 임베딩 계산
        position_embeddings = self.position_embeddings(position_ids)        # 위치 임베딩 계산

        return token_embeddings + position_embeddings                       # 토큰 임베딩과 위치 임베딩을 더하여 최종 임베딩 생성

class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):                        # 클래스의 생성자 메서드로, 필요한 하이퍼파라미터를 인자로 받아 초기화합니다.
        super().__init__()
        self.embedding = Embeddings(                                        # 임베딩 층을 정의하고 초기화합니다. 입력 어휘 사이즈, 임베딩 차원, 최대 길이를 인자로 받습니다.
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(              # 첫 번째 멀티 헤드 어텐션 층을 정의하고 초기화합니다.
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(              # 두 번째 멀티 헤드 어텐션 층을 정의하고 초기화합니다.
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()             # 
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")  # 첫 번째 피드포워드 신경망 층을 정의하고 초기화합니다.
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)                 # 두 번째 피드포워드 신경망 층을 정의하고 초기화합니다.

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size()        # 출력 층을 정의하고 초기화합니다.
                                         , activation="softmax") 

        self.dropout_1 = tf.keras.layers.Dropout(0.3)                       # 첫 번째 드롭아웃 층을 정의하고 초기화합니다.
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    

    def call(self, input_ids, encoder_output, training, mask=None):         # 디코더의 한 층에 대한 계산을 수행하는 메서드입니다. 입력 데이터, 인코더의 출력, 학습 여부, 마스크를 인자로 받습니다.
        embeddings = self.embedding(input_ids)                              # 

        combined_mask = None                                                # 병합된 마스크를 초기화합니다. 조건에 따라 값이 변경됩니다.
        padding_mask = None                                                 # 패딩 마스크를 초기화합니다. 조건에 따라 값이 변경됩니다.
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)        # 
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

class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")


    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    

    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 8)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor = 'val_acc')

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam()
    , loss=cross_entropy
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

img_path = 'D:\\_data\\coco\\archive\\test\\7.jpg'
im = Image.open(img_path)
im.show()

pred_caption = generate_caption(img_path, add_noise=False)
print('Predicted Caption:', pred_caption)

from gtts import gTTS



'''