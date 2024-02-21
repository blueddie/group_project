import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, Add
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.utils import img_to_array, load_img,pad_sequences
import cv2
import numpy as np


video_path = "c:\\_data\\project\\sports\\D3_SP_0728_000001.mp4\\"
frames_dir = "c:\\_data\\project\\save_images\\"


def video_to_frames(video_path, frames_dir, skip_frames=1):
    video = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if count % skip_frames == 0:
            frame_path = f"{frames_dir}/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
        count += 1

    video.release()


# 이미지 전처리
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = preprocess_input(img_array)  # EfficientNetB0에 맞는 전처리
#     img_array = np.expand_dims(img_array, axis=0)  # 모델에 맞게 차원 확장
#     return img_array

# 이미지 특징 추출

def feature_extractor():
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    return base_model
 

# feature_extractor 함수는 EfficientNetB0 모델을 사용하여 이미지의 특징을 추출합니다.
# include_top=False는 네트워크의 최상위 레이어(분류 레이어)를 포함하지 않는다는 것을 의미합니다. 이는 우리가 특징 추출만을 원하기 때문입니다.
# weights='imagenet'는 ImageNet 데이터셋에 대해 사전 훈련된 가중치를 사용하겠다는 것을 의미합니다.
# pooling='avg'는 특징 맵(feature maps)의 글로벌 평균 풀링을 적용합니다.
# layer.trainable = False는 특징 추출기의 파라미터가 훈련 중에 업데이트되지 않도록 설정합니다.

# 이미지 텍스트 캡션 추출
 
class ImageCaptioningModel(Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(ImageCaptioningModel, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)
        self.feature_extractor = feature_extractor()

    def call(self, inputs, training=False):
        img, seq = inputs
        
        # 이미지 특징 추출
        img_features = self.feature_extractor(img)
        img_features = tf.expand_dims(img_features, 1)

        # 캡션 임베딩
        seq_emb = self.embedding(seq)

        # LSTM 입력을 위해 이미지 특징과 캡션 임베딩을 결합
        combined_inputs = tf.concat([img_features, seq_emb], axis=1)

        # LSTM을 통과
        lstm_output, _, _ = self.lstm(combined_inputs)

        # 최종 출력 생성
        output = self.dense(lstm_output)

        return output

# 모델 파라미터 설정
EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = 10000  # 예제 값, 실제 어휘 사전 크기에 맞춰야 함

model = ImageCaptioningModel(VOCAB_SIZE, EMBEDDING_DIM, UNITS)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit


