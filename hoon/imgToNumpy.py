from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

train_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:\\group_project_data\\coco2017\\train2017\\'

st = time.time()

Xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(256, 256),
    batch_size=100,
    color_mode='rgb', # default
    shuffle=True  # 데이터를 무작위로 섞음
)

print('train data ok')

X = []

num_batches = len(Xy_train)
for i in range(num_batches):
    images, labels = Xy_train.next()
    X.append(images)
    print(f"Batch {i+1}/{num_batches} processed")

X = np.concatenate(X, axis=0)

np_path = 'C:\\group_project_data\\coco2017\\numpy\\'

et = time.time()

print(X.shape)

np.save(np_path + 'coco_train.npy', arr=X)
print(f"걸린 시간 : {(et - st)} 초")