import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import get_data

'''
Use following command to run the script
        python main.py

'''

class TripRecommendation(object):
    def __init__(self):
        Inputs, labels = get_data()
        self.X = Inputs
        self.Y = labels
        self.num_classes = len(set(self.Y))
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(dense1, activation='relu', name='dense1')(inputs)
        x = Dense(dense2, activation='relu', name='dense2')(x)
        x = Dense(dense3, activation='relu', name='dense3')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        self.classifier()
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split
                            )
        self.save_model(model_weights)

    def save_model(self ,model_weights):
        self.model.save(model_weights)


if __name__ == "__main__":
    model = TripRecommendation()
    model.train()