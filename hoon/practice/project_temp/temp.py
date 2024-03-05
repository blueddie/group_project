import pygame
import time

# pygame 초기화
pygame.init()

# mp3 파일 로드
pygame.mixer.music.load("c:\\group_project_data\\makeData\\tts.mp3")

# mp3 파일 재생
pygame.mixer.music.play()

# 2초 대기
time.sleep(2)

# pygame 종료
pygame.quit()