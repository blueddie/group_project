from ultralytics import YOLO
import cv2
import numpy as np
from keras.applications import InceptionV3

model = YOLO('yolov8s.pt')

# 동영상 파일 경로
video_path =  "C:\\_data\\project\\003.비디오 장면 설명문 생성 데이터\\01-1.정식개방데이터\\Training\\01.원천데이터\\TS_드라마_220905\\D3_DR_0905_000026.mp4"
cap = cv2.VideoCapture(video_path)

paused = False  # 일시 정지 상태를 추적하는 변수

# 추출된 특성을 저장할 리스트
features_list = []
feature_model = InceptionV3(weights='imagenet', include_top=False)
while cap.isOpened():
    if not paused:
        # 동영상 프레임 읽어오기
        success, frame = cap.read()

        if success:
            # YOLOv8을 사용하여 프레임에서 객체 감지
            results = model(frame)

            # 감지된 객체들에 대해 반복
            for det in results.xyxy[0]:
    # detection이 Results 객체인지 확인
                # if isinstance(detection, list):
        # Results 객체에서 bounding box 좌표 가져오기
                    # x1, y1, x2, y2 = map(int, detection[:4])
                # else:
        # 다른 형식의 결과인 경우 처리
                    # continue
    # bounding box로 객체의 이미지 영역 자르기
                    # object_image = frame[y1:y2, x1:x2]
                    # object_image = cv2.resize(object_image, (299, 299))
    # 여기서 object_image를 특성 추출 모델에 전달하여 특성을 추출
    # 예: object_image를 CNN에 입력하여 특성을 추출하고 저장
                    # features = feature_model.predict(np.expand_dims(object_image, axis=0))
    # 추출된 특성을 features_list에 추가
                    # features_list.append(features.flatten())  # 여기서는 이미지 그대로 저장
                print(det)
        else:
            # 동영상의 끝에 도달하면 루프 종료
            break

    # 'q' 키를 누르면 루프 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # 'p' 키를 누르면 일시 정지/재개
    elif key == ord("p"):
        paused = not paused

# 비디오 캡처 객체 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()

# 추출된 특성을 NumPy 배열로 변환
features_array = np.array([image.flatten() for image in features_list])

np_path = 'C:\\_data\\project\\'

# # NumPy 배열을 파일로 저장
np.save(np_path + 'features.npy', features_array)

# 저장된 특성을 불러오기
# loaded_features = np.load(np_path + 'features.npy')

# print(loaded_features.shape)



