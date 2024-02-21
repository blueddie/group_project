import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, Add,Input, LSTM, Concatenate,RepeatVector
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.utils import img_to_array, load_img,pad_sequences
import cv2
import numpy as np
import os   # 파일 시스템 관리를 위한 os 모듈 임포트
from keras.utils import to_categorical

# video_path = "c:\\_data\\project\\sports\D3_SP_0728_000001.mp4\\"
# output_folder = "c:\\_data\\project\\save_images\\"

def video_to_frames(video_path, skip_frames=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % skip_frames == 0:
            frames.append(frame)
        count += 1
    
    video.release()
    return frames


# 이미지 전처리
def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        # EfficientNetB0에 맞는 크기로 리사이징 및 전처리
        frame = cv2.resize(frame, (224, 224))
        img_array = img_to_array(frame)
        img_array = preprocess_input(img_array)
        processed_frames.append(img_array)
    return np.array(processed_frames)

# 특징 추출기 정의
def feature_extractor():
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

# 비디오 경로
video_path =  "c:\\_data\\project\\sports\\D3_SP_0728_000001.mp4"

# 비디오에서 프레임 추출
frames = video_to_frames(video_path, skip_frames=10)  # 예: 10 프레임마다 하나씩 추출

# 프레임 전처리
processed_frames = preprocess_frames(frames)

# 특징 추출기 로드
model = feature_extractor()

# 특징 추출
features = model.predict(processed_frames)

print("Extracted features shape:", features.shape)

def build_captioning_model(vocab_size, max_length, feature_size):
    # 이미지 특징 입력
    inputs1 = Input(shape=(feature_size,))
    fe1 = Dense(256, activation='relu')(inputs1)
    
    # 시퀀스 모델 입력
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)
    
    # 디코더
    decoder1 = Concatenate()([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # 모델
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# 모델 파라미터 설정 (예시 값)
vocab_size = 10000  # 어휘 사전의 크기
max_length = 34  # 캡션의 최대 길이
feature_size = 1280  # EfficientNetB0의 특징 벡터 크기 (avg pooling 적용 시)

# 캡셔닝 모델 생성
captioning_model = build_captioning_model(vocab_size, max_length, feature_size)

# 예시: 모델 요약 출력
captioning_model.summary()

max_length = 20  # 예시 값

# 모델 구성
embedding_dim = 256
units = 512
vocab_size = 10000  # 예시 값, 실제 어휘 사전 크기에 맞춰야 합니다.

# 이미지 특징 입력
input_features = Input(shape=(features.shape[1],))
fe1 = Dense(units, activation='relu')(input_features)
fe2 = RepeatVector(max_length)(fe1)

# 캡션 입력
input_seqs = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_seqs)
se2 = LSTM(units, return_sequences=True)(se1)

# 디코더
decoder1 = Concatenate(axis=-1)([fe2, se2])
decoder2 = LSTM(units, return_sequences=True)(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# 모델
model = Model(inputs=[input_features, input_seqs], outputs=outputs)

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 학습 데이터 준비
# 가정: `captions`는 전처리된 캡션 데이터, `features`는 추출된 이미지 특징입니다.
# 여기서는 `captions`와 `features`가 이미 준비되어 있다고 가정합니다.

# 캡션에 대한 원-핫 인코딩 및 패딩
# captions_pad = pad_sequences(captions, maxlen=max_length, padding='post')
# captions_oh = to_categorical(captions_pad, num_classes=vocab_size)

# # 모델 학습
# model.fit([features, captions_pad], captions_oh, epochs=20, batch_size=64)

