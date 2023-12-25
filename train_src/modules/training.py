import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import keras
from keras.layers import Input, MaxPooling1D, Dense, Conv1D, Concatenate, GlobalMaxPooling1D, Dropout
from keras.models import Model
import random
from functools import partial
import concurrent.futures

# Module imports
from data_generator import data_generator

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class train_neural_net():
    def __init__(self, EPOCS, LEARNING_RATE):
        self.EPOCS = EPOCS
        self.LEARNING_RATE = LEARNING_RATE

