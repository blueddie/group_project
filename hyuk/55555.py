import cv2
from transformers import pipeline

video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(5) # 비디오의 프레임 레이트 가져오기
while(cap.isOpened()):
    frame_id = cap.get(1) # 현재 프레임 번호
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % (int(frame_rate)*1) == 0: # 초당 1프레임 추출
        filename = "frame%d.jpg" % frame_id
        cv2.imwrite(filename, frame)

cap.release()


# 이미지 캡셔닝 파이프라인 초기화
image_captioning = pipeline(task="image-captioning", model="C:\Users\AIA\anaconda3\envs\tf290gpu\Lib\site-packages\transformers")

# 이미지 파일로부터 설명 생성
image_path = "frame1.jpg" # 앞서 추출한 프레임 중 하나
result = image_captioning(image_path)
caption = result[0]["caption"]

print("Generated caption:", caption)









