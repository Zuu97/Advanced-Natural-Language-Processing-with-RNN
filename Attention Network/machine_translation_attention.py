import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from variables import*
from util import machine_translation_data
from keras.utils.generic_utils import get_custom_objects
import logging
logging.getLogger('tensorflow').disabled = True

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

class Attention(object):
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

    @staticmethod
    def softmax_over_time(x):
        assert(K.ndim(x) > 2)
        numerator = K.exp(x - K.max(x, axis=1, keepdims=True))
        denominator = K.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def training_pase(self):
        # make encoder
        encoder_inputs  = Input(
                                shape=(self.max_length_encoder,),
                                dtype='int32',
                                name='encoder_inputs')
        embedding_encoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_encoder,
                                    input_length=self.max_length_encoder,
                                    name="encoder_embedding"
                                    )(encoder_inputs)

        encoder_output = Bidirectional(
                            LSTM(
                               hidden_dim_encoder,
                               return_sequences=True
                                ),
                            name='bidirectional_lstm_encoder'
                            )(embedding_encoder)
        encoder_output = Dropout(0.5)(encoder_output)
        # make decoder but before that need to build attention so only specify embedding layer
        decoder_inputs = Input(
                            shape=(self.max_length_decoder,),
                            dtype='int32',
                            name='decoder_inputs')

        embedding_decoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_decoder,
                                    input_length=self.max_length_decoder,
                                    name="decoder_embedding"
                                    )(decoder_inputs)

        # Attention
        # we use several layers as global because we use them Ty times for calculate context vector

        repeat_layer = RepeatVector(self.max_length_encoder, name='repeat_vector')
        concat_layer = Concatenate(axis=-1, name='attetion_concat')
        dense1 = Dense(d1, name='dense1_attention', activation='tanh')
        dense2 = Dense(d2, name='dense2_attention', activation=Attention.softmax_over_time)
        dot = Dot(axes=1, name='attention_dot')

        # this function need to loop through Ty times to calculate context
        def attention_step_func(h, st_1):
            st_1 = repeat_layer(st_1) # Copy Tx times (Tx, M2)
            x = concat_layer([h, st_1]) #  (Tx, M2 + 2*M1)
            x = dense1(x)
            alphas = dense2(x)
            context = dot([alphas, h])
            return context

        # decoder continues
        decoder_lstm_layer = LSTM(
                                hidden_dim_decoder,
                                return_state=True,
                                name='lstm_decoder'
                                )
        decoder_dense_layer = Dense(
                                self.vocab_size_decoder,
                                activation='softmax',
                                name='decoder_dense')
        s0 = Input(shape=(hidden_dim_decoder,))
        c0 = Input(shape=(hidden_dim_decoder,))
        context_concat_layer = Concatenate(axis=2, name='contex_concat')

        '''
        for the simplicity here is the psudecode for the loop

        h = encoder(input)
        s, c = 0
        for t in range(Ty):
            alphas = NeuralNet(s, h)
            context = dot(alphas, h)
            o, s = decoder_lstm(context, initial_state = [s, c])
            output+prediction = dense(o)
        '''

        s = s0
        c = c0
        outputs = []
        for t in range(self.max_length_decoder):
            context = attention_step_func(encoder_output, s) # shape (N, 2M1)
            seq_layer = Lambda(lambda x: x[:, t:t+1], name='lambda_layer') # how ever we pass the entire target sequence to decoder we do teacher forcing word by word.
            Xt = seq_layer(embedding_decoder) # shape (N, Tx)

            decoder_lstm_input = context_concat_layer([context, Xt]) # shape (N, 2M1+Tx)
            out, s, c = decoder_lstm_layer(decoder_lstm_input, initial_state=[s, c])
            out = Dropout(0.5)(out)
            decoder_output = decoder_dense_layer(out) # shape (N, V)
            outputs.append(decoder_output) # shape(Ty, N, V)

        # output has shape of (Ty, N, V) but typically batch_size become first so change the dimensions
        def reconfigure_dims(x):
            return K.permute_dimensions(K.stack(x), pattern=(1, 0, 2))

        outputs = Lambda(reconfigure_dims, name='reconfigure_output')(outputs)

        self.attention_network = Model(
            inputs=[
               encoder_inputs,
               decoder_inputs,
               s0,
               c0
            ],
            outputs=outputs
        )

    def train_model(self):
        print("Training Attention Network !!!")
        self.training_pase()
        s0 = np.zeros((len(self.Xencoder_pad), hidden_dim_decoder))
        c0 = np.zeros((len(self.Xencoder_pad), hidden_dim_decoder))
        self.attention_network.compile(
                                    loss = Attention.custom_loss,
                                    optimizer='adam',
                                    metrics=[Attention.acc]
                                        )
        self.attention_network.fit(
                            [self.Xencoder_pad, self.Xdecoder_pad, s0, c0],
                            self.Ydecoder_pad,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=cutoff
                                )
        self.save_model()

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


    def save_model(self):
        model_json = self.attention_network.to_json()
        with open(attention_model_path, "w") as json_file:
            json_file.write(model_json)
        self.attention_network.save_weights(attention_model_weights)
        self.attention_network.compile(
                                    loss = Attention.custom_loss,
                                    optimizer='adam',
                                    metrics=[Attention.acc]
                                        )
    def load_model(self):
        print("Loading Attention Network !!!")
        json_file = open(attention_model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'softmax_over_time': Attention.softmax_over_time})
        loaded_model.load_weights(attention_model_weights)
        self.attention_network = loaded_model

    def testing_pace(self):
        ########################################## E n c o d e r #########################################################
        encoder_inputs  = Input(
                                shape=(self.max_length_encoder,),
                                dtype='int32',
                                name='encoder_inputs')
        embedding_encoder = Embedding(
                                    output_dim=embedding_dim,
                                    input_dim=self.vocab_size_encoder,
                                    input_length=self.max_length_encoder,
                                    name="encoder_embedding"
                                    )(encoder_inputs)

        encoder_output = Bidirectional(
                            LSTM(
                               hidden_dim_encoder,
                               return_sequences=True
                                ),
                                name='bidirectional_lstm_encoder'
                            )(embedding_encoder)
        encoder_output = Dropout(0.5)(encoder_output)
        self.encoder = Model(encoder_inputs, encoder_output)

        self.encoder.get_layer(name='encoder_embedding').set_weights(
                                        self.attention_network.get_layer(name='encoder_embedding').get_weights()
                                        )
        self.encoder.get_layer(name='bidirectional_lstm_encoder').set_weights(
                                        self.attention_network.get_layer(name='bidirectional_lstm_encoder').get_weights()
                                        )

        ########################################## D e c o d e r #########################################################
        decoder_inputs_h = Input(
                            shape=(self.max_length_encoder,hidden_dim_encoder * 2),
                            name='decoder_inputs')

        decoder_input_single = Input(
                            shape=(1,),
                            )

        embedding_decoder_single = Embedding(
                                output_dim=embedding_dim,
                                input_dim=self.vocab_size_decoder,
                                input_length=1,
                                name="decoder_embedding"
                                )(decoder_input_single)

        # Attention
        # we use several layers as global because we use them Ty times for calculate context vector

        repeat_layer = RepeatVector(self.max_length_encoder, name='repeat_vector')
        concat_layer = Concatenate(axis=-1, name='attetion_concat')
        dense1 = Dense(d1, name='dense1_attention', activation='tanh')
        dense2 = Dense(d2, name='dense2_attention', activation=Attention.softmax_over_time)
        dot = Dot(axes=1, name='attention_dot')

        # this function need to loop through Ty times to calculate context
        def attention_step_func(h, st_1):
            st_1 = repeat_layer(st_1) # Copy Tx times (Tx, M2)
            x = concat_layer([h, st_1]) #  (Tx, M2 + 2*M1)
            x = dense1(x)
            alphas = dense2(x)
            context = dot([alphas, h])
            return context

        # decoder continues
        decoder_lstm_layer = LSTM(
                                hidden_dim_decoder,
                                return_state=True,
                                name='lstm_decoder'
                                )
        decoder_dense_layer = Dense(
                                self.vocab_size_decoder,
                                activation='softmax',
                                name='decoder_dense')
        s0 = Input(shape=(hidden_dim_decoder,))
        c0 = Input(shape=(hidden_dim_decoder,))
        context_concat_layer = Concatenate(axis=2, name='contex_concat')

        context = attention_step_func(decoder_inputs_h, s0)
        decoder_lstm_input = context_concat_layer([context, embedding_decoder_single])
        out, s, c = decoder_lstm_layer(decoder_lstm_input, initial_state=[s0, c0])
        out = Dropout(0.5)(out)
        decoder_output = decoder_dense_layer(out)

        self.decoder = Model(
            inputs = [
                decoder_input_single,
                decoder_inputs_h,
                s0,
                c0
            ],
            outputs=[decoder_output,s,c]
        )

        self.decoder.get_layer(name='repeat_vector').set_weights(
                                        self.attention_network.get_layer(name='repeat_vector').get_weights()
                                        )
        self.decoder.get_layer(name='attetion_concat').set_weights(
                                        self.attention_network.get_layer(name='attetion_concat').get_weights()
                                        )
        self.decoder.get_layer(name='dense1_attention').set_weights(
                                        self.attention_network.get_layer(name='dense1_attention').get_weights()
                                        )
        self.decoder.get_layer(name='dense2_attention').set_weights(
                                        self.attention_network.get_layer(name='dense2_attention').get_weights()
                                        )
        self.decoder.get_layer(name='attention_dot').set_weights(
                                        self.attention_network.get_layer(name='attention_dot').get_weights()
                                        )


        self.decoder.get_layer(name='lstm_decoder').set_weights(
                                        self.attention_network.get_layer(name='lstm_decoder').get_weights()
                                        )
        self.decoder.get_layer(name='decoder_dense').set_weights(
                                        self.attention_network.get_layer(name='decoder_dense').get_weights()
                                        )


    def translate_line(self, input_seq, index2word_decoder):
        eos = self.word2idx_decoder['<eos>']
        encoder_output = self.encoder.predict(np.array([input_seq]))

        s = np.zeros((1, hidden_dim_decoder))
        c = np.zeros((1, hidden_dim_decoder))

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx_decoder['<sos>']
        predicted_seq = []
        for _ in range(self.max_length_decoder):
            decoder_out,s,c = self.decoder.predict([target_seq, encoder_output,s,c])
            idx = np.argmax(decoder_out.flatten())
            if idx == eos:
                break
            if idx > 0:
                word = index2word_decoder[idx]
                predicted_seq.append(word)
                target_seq[0, 0] = idx
        return ' '.join(predicted_seq)

    def language_translation(self):
        self.testing_pace()
        index2word_decoder = {idx:word for word,idx in self.word2idx_decoder.items()}

        while True:
            i = np.random.choice(num_samples)
            input_sentence = self.Xencoder[i:i+1][0]
            input_seq = self.Xencoder_pad[i:i+1][0]
            english_sentence = self.translate_line(input_seq, index2word_decoder)

            true_input_sentence = self.Xdecoder[i:i+1][0].split('<sos> ')[1]
            print("True English Sentence: ", true_input_sentence)
            print("Spanish Sentence: ", input_sentence)
            print("Translated English Sentence: ", english_sentence)
            print("-------------------------------------------------------------------------------------------")
            next_one = input("Continue? [Y/n]")
            if next_one and next_one.lower().startswith('n'):
                break

if __name__ == "__main__":
    model = Attention()
    model.tokenize_encoder()
    model.tokenize_decoder()
    get_custom_objects().update({'softmax_over_time': Activation(Attention.softmax_over_time)})
    if not os.path.exists(attention_model_path) or not os.path.exists(attention_model_weights):
        model.train_model()
    model.load_model()
    model.language_translation()