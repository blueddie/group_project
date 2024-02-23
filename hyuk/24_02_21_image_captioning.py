from transformers import AutoTokenizer, ImageToTextPipeline, pipeline
import cv2
from PIL import Image
import numpy as np



model_name = "Salesforce/blip-image-captioning-large"
image_captioning_pipeline = pipeline("image-to-text", model=model_name)

def video_to_frames(video_path, skip_frames=1):
    """
    영상에서 프레임을 추출하는 함수.
    :param video_path: 영상 파일의 경로.
    :param skip_frames: 추출할 프레임 간의 간격.
    :return: 추출된 프레임의 리스트.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

def generate_caption(image, image_captioning_pipeline):
    """
    이미지에 대한 캡션을 생성하는 함수.
    :param image: 이미지 데이터.
    :param image_captioning_pipeline: 이미지 캡셔닝 파이프라인.
    :return: 생성된 텍스트 캡션.
    """
    # 이미지를 PIL Image 객체로 변환
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 파이프라인을 사용해 캡션 생성
    outputs = image_captioning_pipeline(image)
    
    # 생성된 텍스트 반환
    return outputs[0]['generated_text']

# 영상 파일 경로 설정
video_path = "c:\\_data\\project\\sports\\D3_SP_0728_000001.mp4"


# 영상에서 프레임 추출
frames = video_to_frames(video_path, skip_frames=1)

# 첫 번째 프레임에 대한 캡션 생성 (예시)
if frames:
    caption = generate_caption(frames[500], image_captioning_pipeline)
    print("Generated Caption:", caption)
else:
    print("No frames extracted.")


