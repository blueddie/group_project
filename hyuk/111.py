import cv2  # OpenCV 라이브러리 임포트
import os   # 파일 시스템 관리를 위한 os 모듈 임포트

def extract_frames(video_path, output_folder):
    """
    영상 파일에서 프레임을 추출하여 이미지 파일로 저장하는 함수
    :param video_path: 추출할 영상 파일 경로
    :param output_folder: 추출된 프레임을 저장할 폴더 경로
    """
    # 영상 파일 열기
    video_capture = cv2.VideoCapture(video_path)
    # 프레임 카운트 초기화
    frame_count = 0

    # 영상이 열렸는지 확인
    while video_capture.isOpened():
        # 프레임 읽기
        ret, frame = video_capture.read()
        if not ret:  # ret이 False면 영상이 끝난 것이므로 루프를 종료
            break

        # 프레임 저장
        frame_count += 1
        frame_name = f"frame_{frame_count}.jpg"  # 프레임 파일명 생성
        frame_path = os.path.join(output_folder, frame_name)  # 프레임 저장 경로 생성
        cv2.imwrite(frame_path, frame)  # 프레임을 이미지 파일로 저장

    # 영상 파일 닫기
    video_capture.release()
    cv2.destroyAllWindows()

# 영상 파일 경로 설정
video_path = 'C:\\group_project_data\\train\\123.mp4'
# 프레임을 저장할 폴더 설정
output_folder = 'C:\\group_project_data\\test'

# 프레임 추출 실행
extract_frames(video_path, output_folder)