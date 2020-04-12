import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM,Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import model_from_json
from variables import*
from util import machine_translation_data
from keras.models import model_from_json

import logging
logging.getLogger('tensorflow').disabled = True

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM

class MachineTranslation(object):
    def __init__(self):
        inputs, target_inputs, targets = machine_translation_data()
        self.Xencoder = inputs
        self.Xdecoder = target_inputs
        self.Ydecoder = targets

    def tokenize_encoder(self):
        tokenizer = Tokenizer(num_words = vocab_size)
        tokenizer.fit_on_texts(self.Xencoder)

        Xencoder_seq = tokenizer.texts_to_sequences(self.Xencoder)
        self.max_length_encoder = max(len(s) for s in Xencoder_seq)
        self.Xencoder_pad = pad_sequences(Xencoder_seq, maxlen=self.max_length_encoder, padding=padding_type)

        word2idx_encoder = tokenizer.word_index
        self.vocab_size_encoder = min(len(word2idx_encoder), vocab_size) + 1
        self.word2idx_encoder = word2idx_encoder
        self.tokenizer_encoder = tokenizer
        print("Inputs of Encoder Shape : ",self.Xencoder_pad.shape)

    def tokenize_decoder(self):
        xy =  self.Xdecoder + self.Ydecoder
        tokenizer = Tokenizer(num_words = vocab_size, filters='')
        tokenizer.fit_on_texts(xy)

        Xdecoder_seq = tokenizer.texts_to_sequences(self.Xdecoder)
        self.max_length_decoder = max(len(s) for s in Xdecoder_seq)
        self.Xdecoder_pad = pad_sequences(Xdecoder_seq, maxlen=self.max_length_decoder, padding=padding_type)

        Ydecoder_seq = tokenizer.texts_to_sequences(self.Ydecoder)
        Ydecoder_pad = pad_sequences(Ydecoder_seq, maxlen=self.max_length_decoder, padding=padding_type)

        word2idx_decoder = tokenizer.word_index
        self.vocab_size_decoder = min(len(word2idx_decoder), vocab_size) + 1
        self.word2idx_decoder = word2idx_decoder
        self.tokenizer_decoder = tokenizer

        self.Ydecoder_pad = self.oneHot_decoderTargets(Ydecoder_pad)

        print("Inputs of Decoder Shape : ",self.Xdecoder_pad.shape)
        print("One Hot Outputs of Encoder Shape : ",self.Ydecoder_pad.shape)

    def oneHot_decoderTargets(self, Ydecoder_pad):
        decoder_targets_one_hot = np.zeros((len(self.Xencoder), self.max_length_decoder, self.vocab_size_decoder), dtype='float32')
        for i, seq in enumerate(Ydecoder_pad):
            for j, idx in enumerate(seq):
                if idx != 0:
                    decoder_targets_one_hot[i,j,idx] = 1
        return decoder_targets_one_hot

    def teacher_forcing(self):
        input_encoder = Input(shape=(self.max_length_encoder,), dtype='int32', name='encoder_inputs')
        embedding_encoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_encoder,
                                    input_length=self.max_length_encoder,
                                    name="encoder_embedding"
                                    )(input_encoder)
        encoder_out,h,c = LSTM(
                               hidden_dim,
                               return_state=True,
                               name='lstm_encoder'
                               )(embedding_encoder)
        encoder_output_states = [h, c]

        input_decoder = Input(shape=(self.max_length_decoder,), dtype='int32', name='decoder_inputs')
        embedding_decoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_decoder,
                                    input_length=self.max_length_decoder,
                                    name="decoder_embedding"
                                    )(input_decoder)
        decoder_out,_,_ = LSTM(
                                hidden_dim,
                                return_sequences=True,
                                return_state=True,
                                name='lstm_decoder'
                                )(embedding_decoder, initial_state=encoder_output_states)
        decoder_out = Dense(self.vocab_size_decoder, activation='softmax', name='decoder_dense')(decoder_out)

        self.teacher_forcing_model = Model([input_encoder, input_decoder], decoder_out)

    @staticmethod
    def custom_loss(y_true, y_pred):
        mask = K.cast(y_true > 0, dtype='float32')
        out = mask * y_true * K.log(y_pred)
        return -K.sum(out) / K.sum(mask)

    @staticmethod
    def acc(y_true, y_pred):
        targ = K.argmax(y_true, axis=-1)
        pred = K.argmax(y_pred, axis=-1)
        correct = K.cast(K.equal(targ, pred), dtype='float32')

        mask = K.cast(K.greater(targ, 0), dtype='float32')
        n_correct = K.sum(mask * correct)
        n_total = K.sum(mask)
        return n_correct / n_total

    def train_model(self):
        print("Training Teacher Forcing Model !!!")
        self.teacher_forcing()
        self.teacher_forcing_model.compile(
                                        loss = MachineTranslation.custom_loss,
                                        optimizer='adam',
                                        metrics=[MachineTranslation.acc]
                                            )
        self.teacher_forcing_model.fit(
                                        [self.Xencoder_pad, self.Xdecoder_pad],
                                        self.Ydecoder_pad,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_split=cutoff
                                            )
        self.save_model()

    def save_model(self):
        model_json = self.teacher_forcing_model.to_json()
        with open(teacher_forcing_model_path, "w") as json_file:
            json_file.write(model_json)
        self.teacher_forcing_model.save_weights(teacher_forcing_model_weights)
        self.teacher_forcing_model.compile(
                                        loss = MachineTranslation.custom_loss,
                                        optimizer='adam',
                                        metrics=[MachineTranslation.acc]
                                            )
    def load_model(self):
        print("Loading Teacher Forcing Model !!!")
        json_file = open(teacher_forcing_model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(teacher_forcing_model_weights)
        self.teacher_forcing_model = loaded_model

    def testing_encoder(self):
        input_encoder = Input(shape=(1,), dtype='int32', name='encoder_inputs')
        embedding_encoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_encoder,
                                    input_length=1,
                                    name="encoder_embedding"
                                    )(input_encoder)
        encoder_out,h,c = LSTM(
                               hidden_dim,
                               return_state=True,
                               name='lstm_encoder'
                               )(embedding_encoder)
        encoder_output_states = [h, c]
        self.encoder = Model(input_encoder, encoder_output_states)

        self.encoder.layers[1].set_weights(self.teacher_forcing_model.layers[2].get_weights())
        self.encoder.layers[2].set_weights(self.teacher_forcing_model.layers[4].get_weights())

    def testing_decoder(self):
        input_decoder = Input(shape=(1,), name='inputs')
        encoder_h_out = Input(shape=(hidden_dim,))
        encoder_c_out = Input(shape=(hidden_dim,))
        embedding_decoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_decoder,
                                    input_length=1,
                                    name="decoder_embedding"
                                    )(input_decoder)
        decoder_out,h,c = LSTM(
                                hidden_dim,
                                return_sequences=True,
                                return_state=True,
                                name='lstm_decoder'
                                )(embedding_decoder, initial_state=[encoder_h_out, encoder_c_out])
        decoder_out = Dense(self.vocab_size_decoder, activation='softmax', name='decoder_dense')(decoder_out)

        self.decoder = Model([input_decoder, encoder_h_out, encoder_c_out], [decoder_out,h,c])

        self.decoder.layers[1].set_weights(self.teacher_forcing_model.layers[3].get_weights())
        self.decoder.layers[4].set_weights(self.teacher_forcing_model.layers[5].get_weights())
        self.decoder.layers[5].set_weights(self.teacher_forcing_model.layers[6].get_weights())

    def translate_line(self, input_seq, index2word_decoder):
        eos = self.word2idx_decoder['<eos>']
        encoder_output_states = self.encoder.predict(input_seq)
        h, c = encoder_output_states
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx_decoder['<sos>']
        predicted_seq = []
        for _ in range(self.max_length_decoder):
            decoder_out,h,c = self.decoder.predict([target_seq,h,c])
            idx = np.argmax(decoder_out[0, 0, :])
            if idx == eos:
                break
            if idx > 0:
                word = index2word_decoder[idx]
                predicted_seq.append(word)
                target_seq[0, 0] = idx
        return ' '.join(predicted_seq)

    def language_translation(self):
        self.testing_encoder()
        self.testing_decoder()
        index2word_decoder = {idx:word for word,idx in self.word2idx_decoder.items()}

        while True:
            i = np.random.choice(num_samples)
            input_sentence = self.Xencoder[i:i+1][0]
            input_seq = self.Xencoder_pad[i:i+1][0]
            spanish_sentence = self.translate_line(input_seq, index2word_decoder)

            print("Spanish Sentence: ", input_sentence)
            print("English Sentence: ", spanish_sentence)
            print("-------------------------------------------------------------------------------------------")
            next_one = input("Continue? [Y/n]")
            if next_one and next_one.lower().startswith('n'):
                break

if __name__ == "__main__":
    model = MachineTranslation()
    model.tokenize_encoder()
    model.tokenize_decoder()
    if not os.path.exists(teacher_forcing_model_path) or not os.path.exists(teacher_forcing_model_weights):
        model.train_model()
    model.load_model()
    model.language_translation()


# self.decoder.layers[1].set_weights(self.teacher_forcing_model.layers[3].get_weights())
# self.decoder.layers[4].set_weights(self.teacher_forcing_model.layers[5].get_weights())
# self.decoder.layers[5].set_weights(self.teacher_forcing_model.layers[6].get_weights())

# self.encoder.layers[1].set_weights(self.teacher_forcing_model.layers[2].get_weights())
# self.encoder.layers[2].set_weights(self.teacher_forcing_model.layers[4].get_weights())


# self.encoder.get_layer(name='encoder_embedding').set_weights(self.teacher_forcing_model.get_layer(name='encoder_embedding').get_weights())
# self.encoder.get_layer(name='lstm_encoder').set_weights(self.teacher_forcing_model.get_layer(name='lstm_encoder').get_weights())


# self.decoder.get_layer(name='decoder_embedding').set_weights(self.teacher_forcing_model.get_layer(name='decoder_embedding').get_weights())
# self.decoder.get_layer(name='lstm_decoder').set_weights(self.teacher_forcing_model.get_layer(name='lstm_decoder').get_weights())
# self.decoder.get_layer(name='decoder_dense').set_weights(self.teacher_forcing_model.get_layer(name='decoder_dense').get_weights())