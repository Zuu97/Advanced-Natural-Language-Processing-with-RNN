from nltk import word_tokenize
import string
import os
import numpy as np
from variables import*
from sklearn.utils import shuffle
import tensorflow as tf

def remove_punct(line):
    return line.translate(str.maketrans('','',string.punctuation))

def poetry_data():
    inputs = []
    targets = []
    for line in open(text_path):
        line = line.strip()
        if line:
            input_seq = '<sos> ' + line
            target_seq = line + ' <eos>'
            inputs.append(input_seq)
            targets.append(target_seq)
    inputs, targets = shuffle(inputs, targets)
    return inputs, targets