import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input,Bidirectional, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys

from util import get_data
from variables import *

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9975):
            print("\nReached 99.5% train accuracy.So stop training!")
            self.model.stop_training = True

class cnnTextClassifier(object):
    def __init__(self):
        Ytrain, Xtrain, Ytest, Xtest, Yval, Xval = get_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.Xval = Xval
        self.Yval = Yval

    def tokenize_data(self):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(self.Xtrain)
        word2idx = tokenizer.word_index
        print("Actual Vocabulary Size : ",len(word2idx))

        Xtrain_seq = tokenizer.texts_to_sequences(self.Xtrain)
        self.Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=max_length, truncating=trunc_type)

        Xtest_seq  = tokenizer.texts_to_sequences(self.Xtest)
        self.Xtest_pad = pad_sequences(Xtest_seq, maxlen=max_length)

        Xval_seq  = tokenizer.texts_to_sequences(self.Xval)
        self.Xval_pad = pad_sequences(Xval_seq, maxlen=max_length)

    def cnn_model(self):
        inputs = Input(shape=(max_length,), dtype='int32', name='inputs')
        x = Embedding(
                output_dim=embedded_dim,
                input_dim=vocab_size,
                input_length=max_length
                      )(inputs)
        x = Bidirectional(LSTM(lstm_dim, return_sequences=True))(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(num_classes, activation='sigmoid')(x) # texts some times have more than 1 labels and sometimes have no labels so we do seperate 6 sigmoid operations for each text
        self.model = Model(inputs, outputs)

    def train(self):
        callbacks = myCallback()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
                            self.Xtrain_pad,
                            self.Ytrain,
                            batch_size=batch_size,
                            epochs=epoch,
                            validation_data = [self.Xval_pad, self.Yval],
                            callbacks= [callbacks]
                            )
        self.plot_data()

    def load_model(self):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = loaded_model

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def predict(self):
        loss, accuracy = self.model.evaluate(self.Xtest_pad, self.Ytest)
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
    classifier = cnnTextClassifier()
    classifier.tokenize_data()
    if not os.path.exists(model_path) or not os.path.exists(model_weights):
        print("Bulding , Training, Saving !!!")
        classifier.cnn_model()
        classifier.train()
        classifier.save_model()
    else:
        print("Loading, Evaluating !!!")
        classifier.load_model()
    classifier.predict()