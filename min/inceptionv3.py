import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
tf.random.set_seed(777)
np.random.seed(777)
# print(tf.__version__)
from keras.datasets import cifar10
from keras.applications import VGG16, InceptionV3
import time as tm
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

inceptionv3 = InceptionV3(weights='imagenet', include_top=False)

model = inceptionv3
# for layer in model.layers:
#     print(layer.name)

model.summary()


