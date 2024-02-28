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

BASE_PATH = 'd:/_data/coco/archive/coco2017'                                # base 경로 설정

MAX_LENGTH = 40
VOCABULARY_SIZE = 29630
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 20

# 단어 사전 로드
vocab_file_path = 'D:\\_data\\coco\\archive\\vocab_coco.file'
with open(vocab_file_path, 'rb') as f:
    vocabulary = pickle.load(f)

tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH,
    vocabulary=vocabulary
)

word2idx = tf.keras.layers.StringLookup(
    mask_token="",                                                          # 마스크 처리를 하지 않는다? 
    vocabulary=tokenizer.get_vocabulary())                                  # StringLookup 레이어는 텍스트를 정수로 맵핑한다. tokenizer에 있는 단어사전에 맵핑된 정보를 쓴다.

# print(tokenizer.get_vocabulary())
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),  
    invert=True)  

class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))
    

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings

# CNN Encoder 
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
    return cnn_model  

# transformer Encoder
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

        x = self.layer_norm_2(x + attn_output)                              # 
        return x
    

# Transformer Decoder
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
    
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img    
    
# 이미지에 대한 캡션 예측 함수 정의
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

image_augmentation = tf.keras.Sequential(               
    [   tf.keras.layers.RandomFlip("horizontal"),                           # 수평 방향으로 이미지를 무작위로 뒤집습니다.
        tf.keras.layers.RandomRotation(0.2),                                # 이미지를 최대 20도까지 무작위로 회전합니다.
        tf.keras.layers.RandomContrast(0.3),                                # 이미지의 대비를 무작위로 조절합니다.
    ]
)

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


# 모델 정의
encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)
cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

def create_model():
    model = caption_model()
    return model
# 저장된 가중치 불러오기
new_model = create_model()
new_model.load_weights('D:\\_data\\coco\\archive\\caption_model_2.h5')

# 이미지 경로
img_path = 'D:\\_data\\coco\\archive\\test\\test.jpg'

# 캡션 예측
pred_caption = generate_caption(img_path, add_noise=False)
print('Predicted Caption:', pred_caption)
print()

