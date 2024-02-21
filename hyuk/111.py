import cv2
import numpy as np
from cv2 import dnn
import imutils


# OpenCV의 딥러닝 기반 얼굴 감지 모델과 표정 분석 모델을 로드합니다.

prototxt = 'deploy.prototxt.txt'    
caffemodel='res10_300x300_ssd_iter_140000.caffemodel'     
face_model =  cv2.dnn.readNetFromCaffe( prototxt, caffemodel)

# face_model = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
emotion_model = cv2.dnn.readNetFromTensorflow("emotion-ferplus.t7")

# 표정 분석을 위한 라벨을 정의합니다.
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

# 이미지 파일 경로를 지정합니다.
image_path ='d:\\project\\111\\Training\\22\\123.jpeg\\'
# 이미지를 읽어옵니다.
image = cv2.imread(image_path)

# 이미지에서 얼굴을 감지합니다.
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
face_model.setInput(blob)
detections = face_model.forward()

# 감지된 얼굴 영역에 대해 반복합니다.
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # 일정한 신뢰도 이상인 얼굴에 대해서만 처리합니다.
    if confidence > 0.5:
        # 얼굴 영역을 추출합니다.
        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")
        face = image[startY:endY, startX:endX]

        # 얼굴 영역에서 표정을 분석합니다.
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (48, 48), (0, 0, 0), swapRB=True, crop=False)
        emotion_model.setInput(face_blob)
        predictions = emotion_model.forward()
        emotion_index = np.argmax(predictions[0])

        # 결과를 표시합니다.
        text = EMOTIONS[emotion_index]
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# 결과 이미지를 표시합니다.
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(cv2.__version__)