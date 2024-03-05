
    
import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
import PIL
from PIL import Image
import tempModel
import pickle


# 기본 파일 위치
BASE_PATH = 'd:/_data/coco/archive/coco2017'

MAX_LENGTH = 40
VOCABULARY_SIZE = 29630
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 20

tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH
)
tokenizer.set_vocabulary(pickle.load(open('vocab_coco.file', 'rb')))

print(tokenizer)