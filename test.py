import sys
import numpy as np
import csv
import pandas as pd
from cv2 import imread, imwrite, imshow, destroyAllWindows, waitKey
import time
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, UpSampling2D
from keras.models import Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def siamese_net_initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def siamese_net_initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

input_shape = (64,64,1)
left_input = Input(input_shape)
right_input = Input(input_shape)
model = Sequential()
model.add(Conv2D(64, (5,5), activation='relu', input_shape=input_shape, kernel_initializer=siamese_net_initialize_weights))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
model.add(Flatten())
#model.add(Dense(1024, activation='sigmoid', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias, kernel_regularizer=l2(1e-3)))
model.add(Dense(1024, activation='sigmoid', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
model.summary()
encoded_l = model(left_input)
encoded_r = model(right_input)
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=siamese_net_initialize_bias)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
optimizer = Adam(lr = 0.00006)
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
siamese_net.summary()