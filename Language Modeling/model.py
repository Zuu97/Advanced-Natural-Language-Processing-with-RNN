import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
from variables import*
from util import poetry_data
from language import LanguageModel

if __name__ == "__main__":
    language_gen = LanguageModel()
    language_gen.tokenize_data()
    if not os.path.exists(encoder_path) or not os.path.exists(encoder_path):
        language_gen.train_encoder()
    language_gen.load_model()
    language_gen.decoder(encoder_weights)
    language_gen.generate_poetry()

