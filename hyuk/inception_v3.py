import cv2
import numpy as np
import os
import json
from keras.applications import InceptionV3   
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, TimeDistributed
from keras import Sequential   


def video_to_frames(video_path, frames_dir, skip_frames=1):
    """
    영상에서 프레임을 추출하는 함수.
    :param video_path: 영상 파일의 경로.
    :param skip_frames: 추출할 프레임 간의 간격.
    :return: 추출된 프레임의 리스트.
    """
    cap = cv2.VideoCapture(video_path)  # OpenCV의 cv2.VideoCapture 함수를 사용하여 비디오 파일로부터 프레임을 읽기 위한 객체(cap)를 생성
    frame_count = 0                     # 추출한 프레임의 수를 세기 위한 변수 frame_count를 0으로 초기화
    
    while True:                         # 무한 루프 생성
        ret, frame = cap.read()         # 비디오 캡처 객체로부터 한 프레임을 읽어 ret과 frame 변수에 저장
                                        # ret는 프레임 읽기 성공 여부를 나타내는 불리언 값이고, frame은 읽은 프레임의 이미지 데이터
        if not ret:                     # 더 이상 읽을 프레임이 없으면 (ret가 False이면), 루프를 빠져나옴
            break
        skip_frames = int(skip_frames)
        if frame_count % skip_frames == 0:  # 현재 프레임 번호가 skip_frames로 나누어떨어지면 (즉, 지정된 간격에 해당하는 프레임이면), 해당 프레임을 처리
            # 저장할 파일의 이름 설정 (예: frame_0001.jpg)
            frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
            # 프레임 이미지 파일로 저장
            cv2.imwrite(frame_filename, frame)
        frame_count += 1                # 프레임 카운터를 1 증가
    
    # 비디오 캡처 객체 해제
    cap.release()                       # 비디오 캡처 객체를 해제합니다. 이는 모든 자원을 정리하고 비디오 파일을 닫는 데 필요

# 사용 예
video_path = "C:\\_data\\project\\003.비디오 장면 설명문 생성 데이터\\01-1.정식개방데이터\\Training\\01.원천데이터\\TS_드라마_220816\\D3_DR_0816_000001.mp4"

frames_dir = "c:\\_data\\project\\save_images2\\"
video_to_frames(video_path, frames_dir)


def extract_features(frames_dir, model):
    img = image.load_img(frames_dir, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features


# JSON 파일 로딩

# with open('C:\\_data\\project\\003.비디오 장면 설명문 생성 데이터\\01-1.정식개방데이터\\Training\\02.라벨링데이터\\D3_DR_0816_000001.json', 'r', encoding='utf-8-sig') as file:
#     data = json.load(file)
    

inception_model = InceptionV3(weights='imagenet', include_top=False)
model = Model(inputs=inception_model.input, outputs=inception_model.output)

# 프레임에서 특성 추출
frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
features = np.array([extract_features(frame, model) for frame in frame_files])      
    
    
json_path = 'C:\\_data\\project\\003.비디오 장면 설명문 생성 데이터\\01-1.정식개방데이터\\Training\\02.라벨링데이터\\D3_DR_0816_000001.json'  
    
    
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return data
    
    
label_data = load_json_data(json_path)

print(features[0])
print(features.shape)
# (1800, 1, 6, 6, 2048)


'''
feature_shape = (1800, 1, 8, 8, 2048)  # InceptionV3의 특성 출력 형태
flattened_feature_size = np.prod(feature_shape)  # 특성을 flatten한 크기



# RNN 모델 정의
rnn_model = Sequential([
    # TimeDistributed 레이어는 각 시간 단계에서 동일한 Dense 변환을 적용합니다.
    TimeDistributed(Flatten(), input_shape=(None, *feature_shape)),
    LSTM(512, return_sequences=True),
    LSTM(512),
    Dense(256, activation='relu'),
    Dense(1, activation='relu')  # num_classes는 최종 출력 클래스의 수입니다.
])

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rnn_model.fit(features, label_data, epochs=10, batch_size=32, validation_split=0.2)

results = rnn_model.evaluate(features, label_data)
print(results)    
'''   
    
    
    
    
    
    
    