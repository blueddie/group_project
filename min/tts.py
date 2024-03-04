from gtts import gTTS
import os

# 텍스트 입력
text = "a man is swinging a tennis racket at a tennis ball"

# gTTS를 사용하여 음성 파일 생성
tts = gTTS(text=text, lang='en')

tts.save("C:/Users/AIA/Desktop/temp/000000326379.mp3")