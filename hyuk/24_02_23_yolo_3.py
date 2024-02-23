from ultralytics import YOLO
import cv2
import numpy as np
import os






model = YOLO('yolov8s.pt')



# def video_to_frames(video_path, frames_dir, skip_frames=1):
#     """
#     영상에서 프레임을 추출하는 함수.
#     :param video_path: 영상 파일의 경로.
#     :param skip_frames: 추출할 프레임 간의 간격.
#     :return: 추출된 프레임의 리스트.
#     """
#     cap = cv2.VideoCapture(video_path)  # OpenCV의 cv2.VideoCapture 함수를 사용하여 비디오 파일로부터 프레임을 읽기 위한 객체(cap)를 생성
#     frame_count = 0                     # 추출한 프레임의 수를 세기 위한 변수 frame_count를 0으로 초기화
    
#     while True:                         # 무한 루프 생성
#         ret, frame = cap.read()         # 비디오 캡처 객체로부터 한 프레임을 읽어 ret과 frame 변수에 저장
#                                         # ret는 프레임 읽기 성공 여부를 나타내는 불리언 값이고, frame은 읽은 프레임의 이미지 데이터
#         if not ret:                     # 더 이상 읽을 프레임이 없으면 (ret가 False이면), 루프를 빠져나옴
#             break
#         skip_frames = int(skip_frames)
#         if frame_count % skip_frames == 0:  # 현재 프레임 번호가 skip_frames로 나누어떨어지면 (즉, 지정된 간격에 해당하는 프레임이면), 해당 프레임을 처리
#             # 저장할 파일의 이름 설정 (예: frame_0001.jpg)
#             frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
#             # 프레임 이미지 파일로 저장
#             cv2.imwrite(frame_filename, frame)
#         frame_count += 1                # 프레임 카운터를 1 증가
    
#     # 비디오 캡처 객체 해제
#     cap.release()                       # 비디오 캡처 객체를 해제합니다. 이는 모든 자원을 정리하고 비디오 파일을 닫는 데 필요

# # 사용 예
# video_path = "C:\\_data\\project\\003.비디오 장면 설명문 생성 데이터\\01-1.정식개방데이터\\Training\\01.원천데이터\\TS_드라마_220816\\D3_DR_0816_000001.mp4"
# cap = cv2.VideoCapture(video_path)
frames_dir = "c:\\_data\\project\\save_images3\\"
# video_to_frames(video_path, frames_dir)

frame_files = sorted(os.listdir(frames_dir))

for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame_image = cv2.imread(frame_path)  # 프레임 이미지 로드
    detections = model.predict(frame_image)  # YOLO 모델을 사용하여 객체 감지

    for detection in detections:
        class_label = detection['class']  # 감지된 객체의 클래스 라벨
        bbox_coordinates = detection['relative_coordinates']  # 감지된 객체의 바운딩 박스 좌표
        confidence = detection['confidence']








