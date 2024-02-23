
from ultralytics import YOLO
import cv2

model = YOLO('yolov5s.pt')






# results = model.train(data='coco8.yaml' , epochs= 50, imgsz = 640 )
# print(results)
# 동영상 파일 사용시
video_path =  'C:\\_data\\project\\111.mp4'
cap = cv2.VideoCapture(video_path)


paused = False  # 일시 정지 상태를 추적하는 변수

while cap.isOpened():
    if not paused:
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        else:
            # Break the loop if the end of the video is reached
            break

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break
    # Toggle pause if 'p' is pressed
    elif key == ord("p"):
        paused = not paused
        while paused:
            # Wait for another 'p' press to unpause
            if cv2.waitKey(1) & 0xFF == ord("p"):
                paused = not paused
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# print(results[0]['names'])
print(results.names)
