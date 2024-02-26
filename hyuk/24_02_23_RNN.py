import os
import json
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.utils import pad_sequences
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model




json_path =  "c:\\_data\\project\\imageLabelCaption3.json" 
    
    
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return data

data = load_json_data(json_path)

labels = {filename: image_data['labels'] for filename, image_data in data.items()}  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(labels)

x1 = tokenizer.texts_to_sequences(labels)
print(x1)


# print(labels)

captions = {filename: image_data['captions'] for filename, image_data in data.items()}


# print(captions)

all_captions = []

# 각 이미지에 대한 캡션들을 모두 리스트에 추가
for captions_list in captions.values():
    all_captions.extend(captions_list)

# 토크나이저 생성 및 훈련

tokenizer.fit_on_texts(all_captions)

# 각 문장을 시퀀스로 변환
x2 = tokenizer.texts_to_sequences(all_captions)

x2_pad_sequences = pad_sequences(x2, maxlen=37, padding='post')


# print(x)
# print(x_pad_sequences[21])



# 라벨 데이터를 원-핫 인코딩
# num_labels = len(labels)
# one_hot_labels = tf.one_hot(labels, num_labels)


# # Seq2Seq 모델 정의
# latent_dim = 256

# # Encoder
# encoder_inputs = Input(shape=(None,))
# encoder_embedding = Embedding(num_labels, latent_dim)(encoder_inputs)
# encoder = LSTM(latent_dim, return_state=True)
# encoder_outputs, state_h, state_c = encoder(encoder_embedding)
# encoder_states = [state_h, state_c]

# # Decoder
# decoder_inputs = Input(shape=(None,))
# decoder_embedding = Embedding(len(tokenizer.word_index) + 1, latent_dim)
# decoder_embedding_output = decoder_embedding(decoder_inputs)
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)
# decoder_dense = Dense(len(tokenizer.word_index) + 1, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)

# # 모델 정의
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # 모델 컴파일
# model.compile(optimizer='adam', loss='categorical_crossentropy')

# # 모델 훈련
# model.fit([labels, padded_sequences], one_hot_labels, batch_size=64, epochs=10, validation_split=0.2)

# results = model.evaluate(labels, padded_sequences)







