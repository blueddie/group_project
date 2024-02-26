from gtts import gTTS
import os

# 텍스트 입력
text = "안녕하세요, 이민형 바보입니다."

# gTTS를 사용하여 음성 파일 생성
tts = gTTS(text=text, lang='ko')

tts.save("C:\\group_project_data\\makeData\\tts.mp3")