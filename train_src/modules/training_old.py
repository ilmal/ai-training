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

def train_neural_net():
    EPOCHS= 500
    LEARNING_RATE= 0.0000001

    def create_model():

        # MODEL 1 
        # no
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            tf.keras.layers.Dense(2000, activation="leaky_relu"),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # MODEL 2
        # ok, acc at about 53%, can continue before overfit, but uncertain
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            tf.keras.layers.Dense(2000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # MODEL 3
        # overfit quickly, not breaking 50%
        feature_augment = FeatureAugment(500, 2)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            feature_augment, # This will augment the first 500 features
            tf.keras.layers.Dense(2000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # MODEL 4
        # Never started to learn, eventually started overfitting. 
        # for next model: balance training to data to 50% pos/neg, keep same structure
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(200, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(100, activation="leaky_relu"),
            tf.keras.layers.Dense(150, activation="leaky_relu"),
            tf.keras.layers.Dense(50, activation="leaky_relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        #MODEL 5
        
        # model from bing

        # Define the model using the Sequential API
        model = tf.keras.Sequential([
            # Add a reshape layer to convert the input vector into a 2D image with one channel
            tf.keras.layers.Reshape((619, 5, 1), input_shape=(3095,)),
            # Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and same padding
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            # Add a max pooling layer with 2x2 pool size and strides of 2
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            # Add another convolutional layer with 64 filters, 3x3 kernel size, ReLU activation, and same padding
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            # Add another max pooling layer with 2x2 pool size and strides of 2
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            # Add a flatten layer to convert the feature maps into a vector
            tf.keras.layers.Flatten(),
            # Add a dense layer with 128 units and Leaky ReLU activation
            tf.keras.layers.Dense(64, activation='leaky_relu'),
            # Add a dropout layer with 0.35 probability to reduce overfitting
            tf.keras.layers.Dropout(0.35),
            # Add an output layer with 10 units and softmax activation for multi-class classification
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # All previos models have the same problem on guessing more or less the same independant from what the data is,
        # which is logical since most of the data provided is the same, only small changes at the end of the featurelist.
        # Add a weight matrix to inncrease the importance of recent data, which hopefully will force the model to guess in 
        # extremes and not safe out.  


        # MODEL 7
        # the model accuracy started going up as the model loss also rose. The accuracy stoped growing after a while at about 52.9%
        def model_7():
            def exponential_smoothing(data, alpha):
                # data is a tensor of shape (batch_size, time_steps, features)
                # alpha is a scalar between 0 and 1 that controls the smoothing factor
                smoothed_data = tf.TensorArray(tf.float32, size=data.shape[1])
                smoothed_data = smoothed_data.write(0, data[:, 0, :]) # initialize with the first observation
                for i in range(1, data.shape[1]):
                    # apply the smoothing formula: S_t = alpha * Y_t + (1 - alpha) * S_t-1
                    smoothed_data = smoothed_data.write(i, alpha * data[:, i, :] + (1 - alpha) * smoothed_data.read(i-1))
                smoothed_data = smoothed_data.stack() # convert to a tensor of shape (time_steps, batch_size, features)
                smoothed_data = tf.transpose(smoothed_data, perm=[1, 0, 2]) # transpose to match the input shape
                return smoothed_data

            # Define the smoothing factor alpha
            alpha = 0.9 # you can choose any value between 0 and 1

            # Create a smoothing layer
            smoothing_layer = tf.keras.layers.Lambda(lambda x: exponential_smoothing(x, alpha))

            # Define your feature vector shape
            feature_shape = (3095,)
            # Define the number of filters and kernel size for each part
            kernel_size1 = 3

            # Create input layers for each part
            input_1 = Input(shape=(495,1))
            input_2 = Input(shape=(2600,1))

            smoothed_input_1 = smoothing_layer(input_1)
            smoothed_input_2 = smoothing_layer(input_2)

            # Part1 layers
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='relu')(input_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='relu')(conv_1)
            conv_1 = Conv1D(filters=128, kernel_size=16, activation='relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='relu')(conv_1)
            pool_1 = GlobalMaxPooling1D()(conv_1)

            # Part2 layers
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='relu')(input_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='relu')(conv_2)
            conv_2 = Conv1D(filters=128, kernel_size=16, activation='relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='relu')(conv_2)
            pool_2 = GlobalMaxPooling1D()(conv_2)


            # Concatenate the outputs
            concat = Concatenate()([pool_1, pool_2])

            # Add a fully connected dense layer
            dense = Dense(128, activation='relu')(concat)
            dense = Dense(128, activation='relu')(dense)
            dense = Dense(128, activation='relu')(dense)
            dense = Dense(128, activation='relu')(dense)
            dense = Dense(64, activation='relu')(dense)
            dense = Dense(64, activation='relu')(dense)
            dense = Dense(64, activation='relu')(dense)
            dense = Dense(64, activation='relu')(dense)
            dense = Dense(32, activation='relu')(dense)
            dense = Dense(32, activation='relu')(dense)


            # Add an output layer for your task (e.g. binary classification)
            output = Dense(1, activation='sigmoid')(dense)

            # Create a model from the inputs and outputs
            model = Model(inputs=[input_1, input_2], outputs=output)
            return model
        model = model_7()


        # MODEL 8, like model 7 but bigger, also removed the exponential smoothing.
        # The accuracy didn't change at all and the loss started rising before any change to the accuracy
        def model_8():
            def exponential_smoothing(data, alpha):
                # data is a tensor of shape (batch_size, time_steps, features)
                # alpha is a scalar between 0 and 1 that controls the smoothing factor
                smoothed_data = tf.TensorArray(tf.float32, size=data.shape[1])
                smoothed_data = smoothed_data.write(0, data[:, 0, :]) # initialize with the first observation
                for i in range(1, data.shape[1]):
                    # apply the smoothing formula: S_t = alpha * Y_t + (1 - alpha) * S_t-1
                    smoothed_data = smoothed_data.write(i, alpha * data[:, i, :] + (1 - alpha) * smoothed_data.read(i-1))
                smoothed_data = smoothed_data.stack() # convert to a tensor of shape (time_steps, batch_size, features)
                smoothed_data = tf.transpose(smoothed_data, perm=[1, 0, 2]) # transpose to match the input shape
                return smoothed_data

            # Define the smoothing factor alpha
            alpha = 0 # you can choose any value between 0 and 1

            # Create a smoothing layer
            smoothing_layer = tf.keras.layers.Lambda(lambda x: exponential_smoothing(x, alpha))

            # Define your feature vector shape
            feature_shape = (3095,)
            # Define the number of filters and kernel size for each part
            kernel_size1 = 3

            # Create input layers for each part
            input_1 = Input(shape=(495,1))
            input_2 = Input(shape=(2600,1))

            smoothed_input_1 = smoothing_layer(input_1)
            smoothed_input_2 = smoothing_layer(input_2)

            # Part1 layers
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            pool_1 = GlobalMaxPooling1D()(conv_1)

            # Part2 layers
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            pool_2 = GlobalMaxPooling1D()(conv_2)


            # Concatenate the outputs
            concat = Concatenate()([pool_1, pool_2])

            # Add a fully connected dense layer
            dense = Dense(128, activation='leaky_relu')(concat)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(32, activation='leaky_relu')(dropout)
            dense = Dense(32, activation='leaky_relu')(dense)


            # Add an output layer for your task (e.g. binary classification)
            output = Dense(1, activation='sigmoid')(dense)

            # Create a model from the inputs and outputs
            model = Model(inputs=[input_1, input_2], outputs=output)
            return model
        model = model_8()

        # MODEL 9, like model 8but with exponential smoothing and extra dropout.
        # the model flatlined and didnt move, no overfitting, just flatline
        def model_9():
            def exponential_smoothing(data, alpha):
                # data is a tensor of shape (batch_size, time_steps, features)
                # alpha is a scalar between 0 and 1 that controls the smoothing factor
                smoothed_data = tf.TensorArray(tf.float32, size=data.shape[1])
                smoothed_data = smoothed_data.write(0, data[:, 0, :]) # initialize with the first observation
                for i in range(1, data.shape[1]):
                    # apply the smoothing formula: S_t = alpha * Y_t + (1 - alpha) * S_t-1
                    smoothed_data = smoothed_data.write(i, alpha * data[:, i, :] + (1 - alpha) * smoothed_data.read(i-1))
                smoothed_data = smoothed_data.stack() # convert to a tensor of shape (time_steps, batch_size, features)
                smoothed_data = tf.transpose(smoothed_data, perm=[1, 0, 2]) # transpose to match the input shape
                return smoothed_data

            # Define the smoothing factor alpha
            alpha = 0.8 # you can choose any value between 0 and 1

            # Create a smoothing layer
            smoothing_layer = tf.keras.layers.Lambda(lambda x: exponential_smoothing(x, alpha))

            # Define your feature vector shape
            feature_shape = (3095,)
            # Define the number of filters and kernel size for each part
            kernel_size1 = 3

            # Create input layers for each part
            input_1 = Input(shape=(495,1))
            input_2 = Input(shape=(2600,1))

            smoothed_input_1 = smoothing_layer(input_1)
            smoothed_input_2 = smoothing_layer(input_2)

            # Part1 layers
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            pool_1 = GlobalMaxPooling1D()(conv_1)

            # Part2 layers
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            pool_2 = GlobalMaxPooling1D()(conv_2)


            # Concatenate the outputs
            concat = Concatenate()([pool_1, pool_2])

            # Add a fully connected dense layer
            dense = Dense(128, activation='leaky_relu')(concat)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dropout = Dropout(0.25)(dense)
            dense = Dense(32, activation='leaky_relu')(dropout)
            dense = Dense(32, activation='leaky_relu')(dense)


            # Add an output layer for your task (e.g. binary classification)
            output = Dense(1, activation='sigmoid')(dense)

            # Create a model from the inputs and outputs
            model = Model(inputs=[input_1, input_2], outputs=output)
            return model
        model = model_9()

        # MODEL 10, like model 9 but without extra dropout.
        # starts to overfit before accuracy has moved, acc about 52.61%
        def model_10():
            def exponential_smoothing(data, alpha):
                # data is a tensor of shape (batch_size, time_steps, features)
                # alpha is a scalar between 0 and 1 that controls the smoothing factor
                smoothed_data = tf.TensorArray(tf.float32, size=data.shape[1])
                smoothed_data = smoothed_data.write(0, data[:, 0, :]) # initialize with the first observation
                for i in range(1, data.shape[1]):
                    # apply the smoothing formula: S_t = alpha * Y_t + (1 - alpha) * S_t-1
                    smoothed_data = smoothed_data.write(i, alpha * data[:, i, :] + (1 - alpha) * smoothed_data.read(i-1))
                smoothed_data = smoothed_data.stack() # convert to a tensor of shape (time_steps, batch_size, features)
                smoothed_data = tf.transpose(smoothed_data, perm=[1, 0, 2]) # transpose to match the input shape
                return smoothed_data

            # Define the smoothing factor alpha
            alpha = 0.8 # you can choose any value between 0 and 1

            # Create a smoothing layer
            smoothing_layer = tf.keras.layers.Lambda(lambda x: exponential_smoothing(x, alpha))

            # Define your feature vector shape
            feature_shape = (3095,)
            # Define the number of filters and kernel size for each part
            kernel_size1 = 3

            # Create input layers for each part
            input_1 = Input(shape=(495,1))
            input_2 = Input(shape=(2600,1))

            smoothed_input_1 = smoothing_layer(input_1)
            smoothed_input_2 = smoothing_layer(input_2)

            # Part1 layers
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            conv_1 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_1)
            pool_1 = GlobalMaxPooling1D()(conv_1)

            # Part2 layers
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(input_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=256, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=128, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            conv_2 = Conv1D(filters=64, kernel_size=16, activation='leaky_relu')(conv_2)
            pool_2 = GlobalMaxPooling1D()(conv_2)


            # Concatenate the outputs
            concat = Concatenate()([pool_1, pool_2])

            # Add a fully connected dense layer
            dense = Dense(128, activation='leaky_relu')(concat)
            dense = Dense(128, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dense = Dense(128, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(128, activation='leaky_relu')(dropout)
            dense = Dense(128, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dense = Dense(64, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dense = Dense(64, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(64, activation='leaky_relu')(dropout)
            dense = Dense(64, activation='leaky_relu')(dense)
            dropout = Dropout(0.20)(dense)
            dense = Dense(32, activation='leaky_relu')(dropout)
            dense = Dense(32, activation='leaky_relu')(dense)


            # Add an output layer for your task (e.g. binary classification)
            output = Dense(1, activation='sigmoid')(dense)

            # Create a model from the inputs and outputs
            model = Model(inputs=[input_1, input_2], outputs=output)
            return model
        model = model_10()

        # MODEL 11
        # going back to first model, more nodes and also more dropout
        # val_acc is constant and not moving, val_loss is increasing and train_loss and train_data is behaving good
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            tf.keras.layers.Dense(2000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(250, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # MODEL 12
        # Rerunning model 2 (overfit) ran without saving data
        # Run with more dropout (more dropout 0.5 leads to flatlines)
        # Run with dropout 0.3 better max 52.94 accuracy,
        # Run with dropout 0.4, worse flatline, and when start moving its a crash (both acc and loss crash)
        # Run with dropout 0.3 and normal rely activation, nope, not working
        # Run with dropout 0.1 and leaky activation, bad barely training
        # Run with dropout 0.1 and multiply layer length with factor of 10, 
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3095,)),
            tf.keras.layers.Dense(15000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(5000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2500, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1000, activation="leaky_relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # MODEL 13
        # I want to try to create a enormous simple network with super high dropout, so thousands of nodes with .9 or .95 dropout each
        # Split the data exactly 50/50

        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy")
            ]
        )

        return model

    def neural_nets(EPOCHS):

        model = create_model()
        #model = keras.models.load_model("./checkpoints")


        val_data_list = os.listdir("./get_training_data_app/dev/output_val/labels/")
        data_list = os.listdir("./get_training_data_app/dev/output_train/labels/")

        val_data_list = os.listdir("/mnt/external/app/AI-project/get_training_data_app/dev/output_val/labels/")
        data_list = os.listdir("/mnt/external/app/AI-project/get_training_data_app/dev/output_train/labels/")

        val_data_dir = "./get_training_data_app/dev/output_val/"
        data_dir = "./get_training_data_app/dev/output_train/"

        val_data_dir = "/mnt/external/app/AI-project/get_training_data_app/dev/output_val/"
        data_dir = "/mnt/external/app/AI-project/get_training_data_app/dev/output_train/"


        #val_generator = DataGenerator(val_data_list, val_data_dir)
        #generator = DataGenerator(data_list, data_dir)

        generator = data_generator(data_dir, 1000)
        val_generator = data_generator(val_data_dir, 1000)

        val_generator_dataset = tf.data.Dataset.from_generator(
            lambda: val_generator, output_signature=(
                tf.TensorSpec(shape=(None, 3095), dtype=tf.float32),
                tf.TensorSpec(shape=(None), dtype=tf.int64)
            ))

        generator_dataset = tf.data.Dataset.from_generator(
            lambda: generator, output_signature=(
                tf.TensorSpec(shape=(None, 3095), dtype=tf.float32),
                tf.TensorSpec(shape=(None), dtype=tf.int64)
            ))

        CACHE_PATH = "./cache/"
        VAL_CACHE_PATH = "./cache_val/"

        val_generator_dataset = val_generator_dataset.cache(VAL_CACHE_PATH + "tf_cache.tfcache").shuffle(100)

        generator_dataset = generator_dataset.cache(CACHE_PATH + "tf_cache.tfcache").shuffle(100)

        generator_dataset = generator_dataset.prefetch(tf.data.AUTOTUNE)

        CHECKPOINT_PATH = "checkpoints/"
        steps_per_epoch = len(data_list)
        SAVE_INTERVAL= 2

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
           filepath=CHECKPOINT_PATH,
           save_weights_only=False,
           save_freq=int(steps_per_epoch * SAVE_INTERVAL)) # 154485

        tensorboard_callback = tf.keras.callbacks.TensorBoard( 
            log_dir=("logs"),
            histogram_freq=1, 
            write_steps_per_second=True
        )

        # Change learning rate
        # keras.backend.set_value(model.optimizer.learning_rate, 0.00001)

        # balance dataset
        #generator_dataset = balance_dataset(generator_dataset)

        model.summary()

        # # Define a function that splits the feature vector
        # def split_feature_vector(feature_vector, label):
        #     # Split the feature vector into two parts
        #     part1, part2 = tf.split(feature_vector, [495, 2600], axis=1)
        #     # Add a dummy dimension to each part
        #     part1 = tf.expand_dims(part1, axis=-1)
        #     part2 = tf.expand_dims(part2, axis=-1)
        #     # Return the parts and the label as a tuple
        #     return (part1, part2), label

        # # Apply the split function to the generator_dataset
        # split_generator_dataset = generator_dataset.map(split_feature_vector)

        # # Apply the split function to the val_generator_dataset
        # split_val_generator_dataset = val_generator_dataset.map(split_feature_vector)

        model.fit(generator_dataset,
                validation_data=val_generator_dataset,
                epochs=1000,
                callbacks=[tensorboard_callback, model_checkpoint_callback],
                initial_epoch=0
        )
        print("saving model")
        model.save("./saved_model_new/model")


    neural_nets(EPOCHS)

    return

if __name__ == "__main__":
    train_neural_net()
