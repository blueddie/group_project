## 이미지에서 얼굴만 잘라내기
import cv2
import numpy as np
import os 

path_dir = 'd:\\project\\111\\Training\\22\\'
file_list = os.listdir(path_dir)

print(file_list[0])
print(len(file_list))

# 확장자명 제거 후 배열 생성
file_name_list = []
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
print(file_list[0])

# image = cv2.imread('.jpg')
# face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_defualt.xml') # 어쩌구 xml은 학습된 얼굴 인식 모델 파일
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 자르기 전에 얼굴을 잘 인식하는지 확인하는 코드
# image변수에 cv2.imread()로 불러오기
# face_cascade 에 cv2.CascadeClassifier() 로 얼굴인식 모델 불러오기
# faces 변수에 .detectMultiScale()로 얼굴 검출
# for (x,y,w,h) in faces:
#     cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
#     cv2.imshow("face_recognition", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # 얼굴을 잘라서 사이즈 조정해주는 코드
# # cropped 변수에 얼굴영역을 지정하고 180*180으로 재조정
# for (x,y,w,h) in faces:
#     cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
#     cropped = image[y:y+h, x:x+w]
#     resize = cv2.resize(cropped, (180,180))
#     cv2.imshow("crop&resize", resize)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



###### 이 위는 테스트. 아래가 최종

def Cutting_face_save(image, name):
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for (w,y,w,h) in faces:
        cropped = image[y:y+h, x:x+w]
        resize = cv2.resize(cropped, (180,180))
        cv2.imwrite(f"D:\project\111\Training\11\{name}.jpg", resize)

for name in file_name_list:
    img = cv2.imread("images/faces/"+name+".jpg")
    Cutting_face_save(img, name)

