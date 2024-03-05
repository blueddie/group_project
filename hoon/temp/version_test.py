import sys
import tensorflow as tf
import keras

# Python 버전 확인
if sys.version_info < (3, 7):
    print("Python 3.7 이상이 필요합니다.")
else:
    print("Python 버전:", sys.version)

# TensorFlow 버전 확인
if tf.__version__ < "2.7.0":
    print("TensorFlow 2.7 이상이 필요합니다.")
else:
    print("TensorFlow 버전:", tf.__version__)

# Keras 버전 확인
if keras.__version__ < "2.7.0":
    print("Keras 2.7 이상이 필요합니다.")
else:
    print("Keras 버전:", keras.__version__)