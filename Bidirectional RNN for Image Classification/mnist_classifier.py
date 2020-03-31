import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Lambda, Concatenate, Input,Bidirectional, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys

from util import get_data
from variables import*

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9975):
            print("\nReached 99.5% train accuracy.So stop training!")
            self.model.stop_training = True

class MnistClassifier(object):
    def __init__(self):
        Xtrain, Ytrain, Xtest , Ytest = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest

    def brnn(self):
        inputs = Input(shape=(image_size,image_size), dtype='float32', name='inputs')

        brnn1  = Bidirectional(LSTM(lstm_dim, return_sequences=True))(inputs)
        x1 = GlobalMaxPooling1D()(brnn1)

        inputs_ = Lambda(lambda t: K.permute_dimensions(t, pattern=(0,2,1)))(inputs)
        brnn2  = Bidirectional(LSTM(lstm_dim, return_sequences=True))(inputs_)
        x2 = GlobalMaxPooling1D()(brnn2)

        concat = Concatenate(axis=1)
        x = concat([x1,x2])
        self.model = Model(inputs, x)

    def train(self):
        callbacks = myCallback()
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=batch_size,
                            epochs=num_epochs,
                            validation_data = [self.Xtest, self.Ytest],
                            callbacks= [callbacks]
                            )
        self.plot_data()

    def load_model(self):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = loaded_model

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def predict(self):
        loss, accuracy = self.model.evaluate(self.Xtest, self.Ytest)
        print("Val_loss: ",loss)
        print("Val_accuracy: ",accuracy)

    def plot_data(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']

        plt.plot(loss, label="Train loss")
        plt.plot(val_loss, label="Validation loss")
        plt.legend()
        plt.show()

        plt.plot(accuracy, label="Train accuracy")
        plt.plot(val_accuracy, label="Validation accuracy")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    classifier = MnistClassifier()
    if not os.path.exists(model_path) or not os.path.exists(model_weights):
        print("Bulding , Training, Saving !!!")
        classifier.brnn()
        classifier.train()
        classifier.save_model()
    else:
        print("Loading, Evaluating !!!")
        classifier.load_model()
    classifier.predict()