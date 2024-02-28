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
from keras.applications import InceptionV3

model = InceptionV3(include_top=False, weights='imagenet')
model.summary()