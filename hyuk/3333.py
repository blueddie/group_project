import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split 
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
import time
from sklearn.preprocessing import OneHotEncoder

# 이미지 파일들이 있는 디렉토리 경로
directories = ['d:\\project\\111\\Training\\22\\']

images = []
labels = []
start = time.time()
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):  
            # 이미지 파일 경로
            image_path = os.path.join(directory, filename)
            
            # 이미지 파일 읽어오기
            image = cv2.imread(image_path)
            
            if image is not None:
                # 이미지를 성공적으로 읽었을 경우, 크기 조정 및 전처리
                image = cv2.resize(image, (300, 300))  # 크기 조정
                image = image.astype('float32') / 255.0  # 정규화
                
                # 이미지와 레이블 추가
                images.append(image)
                labels.append(directory.split('_')[0])  # 디렉토리 이름에서 클래스 레이블 추출
            
images = np.array(images)
labels = np.array(labels)


print(images.shape) 
print(labels.shape)

labels = labels.reshape(-1, 1)
ohe = OneHotEncoder()
y = ohe.fit_transform(labels)




X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=3)



model = Sequential()
model.add(Conv2D(19, (3,3), activation='swish', input_shape=(300, 300, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(97, (4,4), activation='swish'))
model.add(MaxPooling2D())
model.add(Conv2D(9, (3,3), activation='swish'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(21,activation='swish'))
model.add(Dense(1, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=1)
end = time.time()
# 4. 평가, 훈련
loss = model.evaluate(X_test, y_test)
print("ACC : ", loss[1])
print("걸린시간 : ", round(end-start,3),"초")




model.save("c:\\_data\\_save\\project_practice.h5")











