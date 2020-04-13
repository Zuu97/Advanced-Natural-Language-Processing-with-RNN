from nltk import word_tokenize
import string
import os
import numpy as np
from variables import*
from sklearn.utils import shuffle

def machine_translation_data():
    inputs = []
    target_inputs = []
    targets = []
    for i,line in enumerate(open(text_path, encoding="utf8")):
        line = line.strip()
        if line:
            if '\t' in line:
                input_seq = line.split('\t')[1]
                trans_seq = line.split('\t')[0]
                target_input_seq = '<sos> ' + trans_seq
                target_seq = trans_seq + ' <eos>'

                inputs.append(input_seq)
                target_inputs.append(target_input_seq)
                targets.append(target_seq)
                if i >= num_samples - 1:
                    break
    inputs, target_inputs, targets = shuffle(inputs, target_inputs, targets)
    return inputs, target_inputs, targets