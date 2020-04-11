import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM,Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import model_from_json
from variables import*
from util import poetry_data

class LanguageModel(object):
    def __init__(self):
        inputs, targets = poetry_data()
        self.X = inputs
        self.Y = targets

    def tokenize_data(self):
        xy =  self.X + self.Y
        tokenizer = Tokenizer(num_words = vocab_size, filters='', oov_token=oov)
        tokenizer.fit_on_texts(xy)

        X_seq = tokenizer.texts_to_sequences(self.X)
        self.X_pad = pad_sequences(X_seq, maxlen=max_length, truncating=truncating_type)

        Y_seq = tokenizer.texts_to_sequences(self.Y)
        Y_pad = pad_sequences(Y_seq, maxlen=max_length, truncating=truncating_type)

        self.tokenizer = tokenizer
        word2idx = tokenizer.word_index
        self.vocab_size = max(len(word2idx), vocab_size)
        self.word2idx = word2idx

        self.Y_pad = to_categorical(Y_pad, num_classes=self.vocab_size)

    def encoder(self):
        inputs = Input(shape=(max_length,), dtype='int32', name='input')
        c0 = Input(shape=(hidden_dim,))
        h0 = Input(shape=(hidden_dim,))
        emb = Embedding(output_dim=embedding_dim, input_dim=self.vocab_size, input_length=max_length)(inputs)
        x,_,_ = LSTM(hidden_dim, return_sequences=True, return_state=True)(emb, initial_state=[h0, c0])
        x = Dense(self.vocab_size, activation='softmax')(x)

        self.model_encoder = Model([inputs, h0, c0], x)

    def train_encoder(self):
        print("Training Encoder !!!")
        self.encoder()
        self.model_encoder.compile(
                    loss = 'categorical_crossentropy',
                    optimizer=Adam(0.01),
                    metrics=['accuracy']
                          )

        h = np.zeros((len(self.X_pad), hidden_dim))
        c = np.zeros((len(self.X_pad), hidden_dim))
        self.model_encoder.fit(
            [self.X_pad, h, c],
            self.Y_pad,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=cutoff
        )
        self.save_model()

    def save_model(self):
        model_json = self.model_encoder.to_json()
        with open(encoder_path, "w") as json_file:
            json_file.write(model_json)
        self.model_encoder.save_weights(encoder_weights)

    def load_model(self):
        print("Loading Encoder !!!")
        json_file = open(encoder_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(encoder_weights)
        loaded_model.compile(
                    loss = 'categorical_crossentropy',
                    optimizer=Adam(0.01),
                    metrics=['accuracy']
                          )
        self.model_encoder = loaded_model

    def decoder(self, encoder_weights):
        emb_layer = self.model_encoder.layers[1]
        lstm_layer = self.model_encoder.layers[-2]
        dense_layer = self.model_encoder.layers[-1]

        inputs = Input(shape=(1,))
        c0 = Input(shape=(hidden_dim,))
        h0 = Input(shape=(hidden_dim,))
        emb = Embedding(output_dim=embedding_dim, input_dim=self.vocab_size, input_length=1)(inputs)
        x,h,c = LSTM(hidden_dim, return_sequences=True, return_state=True)(emb, initial_state=[h0, c0])
        x = Dense(self.vocab_size, activation='softmax')(x)

        self.model_decoder = Model([inputs, h0, c0], [x,h,c])

        self.model_decoder.layers[1].set_weights(self.model_encoder.layers[1].get_weights())
        self.model_decoder.layers[-2].set_weights(self.model_encoder.layers[-2].get_weights())
        self.model_decoder.layers[-1].set_weights(self.model_encoder.layers[-1].get_weights())
        print("Decoder compiled sucessfully !!!")

        self.model_decoder.compile(
                    loss = 'categorical_crossentropy',
                    optimizer=Adam(0.01),
                    metrics=['accuracy']
                          )

    def generate_line(self):
        eos = self.word2idx['<eos>']
        sos = self.word2idx['<sos>']
        start = np.array([sos])
        idx2word = {index:word for word, index in self.word2idx.items()}
        h = np.zeros((1, hidden_dim))
        c = np.zeros((1, hidden_dim))

        line = []
        input_idx = start
        for _ in range(max_length):
            out,h,c = self.model_decoder.predict([input_idx,h,c])
            probs = out.squeeze()
            probs[0] = 0
            probs = probs / probs.sum()
            next_idx = np.random.choice(len(idx2word), p=probs)
            line.append(idx2word[next_idx])
            input_idx = np.array([next_idx])
            if next_idx == eos:
                break
        return ' '.join(line[:-1])

    def generate_poetry(self, n_lines=3):
        for _ in range(n_lines):
            print(self.generate_line())

