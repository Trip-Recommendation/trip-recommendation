import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import get_data
import joblib

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
        self.label_encoder = joblib.load(label_encoder_weights)
        self.hotel_dict = joblib.load(hotel_dict_path)

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(dense1, activation='relu', name='dense1')(inputs)
        x = Dense(dense2, activation='relu', name='dense2')(x)
        x = Dense(dense3, activation='relu', name='dense3')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        self.model = Model(inputs, outputs)

    def train(self, verbose):
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
        self.model.fit(
                self.X,
                self.Y,
                batch_size=batch_size,
                epochs=num_epoches,
                validation_split=validation_split,
                verbose=verbose
                )

    def finetune(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = inputs
        for layer in self.model.layers[1:-1]:
            layer.trainable = False
            x = layer(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)

        self.model = Model(inputs, outputs)

    def save_model(self):
        self.model.save(model_weights)

    def load_model(self):
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        )
        self.model = loaded_model

    def run(self, verbose=1):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.classifier()
            self.train(verbose)
            self.save_model()

    def run_finetune(self, verbose=0):
        self.load_model()
        self.finetune()
        self.train(verbose)
        self.save_model()

    def prediction(self, user_id):
        user_id = int(user_id)
        try:
            input_ = np.array([self.X[user_id,:]])
        except:
            IndexError

        P = self.model.predict(input_).squeeze()
        Pred = np.argsort(P)[-n_recommendation:]

        labels = self.label_encoder.inverse_transform(Pred)
        return [str(self.hotel_dict[label]) for label in labels]