import cv2
from keras.models import load_model
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 사전 훈련된 얼굴 인식 및 표정 분석 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('your_emotion_model.h5')  # 여기서 'your_emotion_model.h5'는 표정 분석 모델의 파일 경로입니다.

# 표정 레이블 정의
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 이미지 불러오기
img = cv2.imread('path_to_your_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지에서 얼굴 인식
faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

for (x, y, w, h) in faces:
    # 인식된 얼굴 영역 추출 및 크기 조정
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # 표정 분석 모델을 위한 입력 형태로 변환
    roi = roi_gray.astype('float')/255.0  # 정규화
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    # 표정 예측
    prediction = emotion_model.predict(roi)
    max_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[max_index]

    # 예측된 표정과 얼굴 영역을 이미지에 표시
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 결과 이미지 표시
cv2.imshow('Emotion Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()