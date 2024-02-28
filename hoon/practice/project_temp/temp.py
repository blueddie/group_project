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
import joblib
from joblib import dump, load
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle
import time








def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    
    output = inception_v3.output                                           
    output = tf.keras.layers.Reshape(                                       
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)           
    return cnn_model    









