from ultralytics import YOLO
import cv2

# YOLO 모델을 불러옵니다
model = YOLO('yolov8s.pt')

# 동영상 파일 경로
video_path =  "C:\\group_project_data\\coco2017\\train2017\\train\\000000000009.jpg"
cap = cv2.VideoCapture(video_path)

paused = False  # 일시 정지 상태를 추적하는 변수

while cap.isOpened():
    if not paused:
        # 동영상으로부터 한 프레임을 읽어옵니다
        success, frame = cap.read()

        if success:
            # YOLOv8로 프레임에서 객체를 감지합니다
            results = model(frame)

            # 결과를 프레임에 시각화합니다
            annotated_frame = results[0].plot()

            # 시각화된 프레임을 보여줍니다
            cv2.imshow("YOLOv8 감지 결과", annotated_frame)
        else:
            # 동영상의 끝에 도달하면 루프를 탈출합니다
            # break
            pass

    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    
    # 'q'가 눌리면 루프를 탈출합니다
    if key == ord("q"):
        break
    # 'p'가 눌리면 일시 정지/재개합니다
    elif key == ord("p"):
        paused = not paused
        while paused:
            # 다시 'p'가 눌릴 때까지 대기합니다
            if cv2.waitKey(1) & 0xFF == ord("p"):
                paused = not paused

# 동영상 캡처 객체를 해제하고 디스플레이 창을 닫습니다
cap.release()
cv2.destroyAllWindows()
