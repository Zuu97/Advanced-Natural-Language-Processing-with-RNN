import os
embedded_dim = 100
max_length = 120
vocab_size = 20000
oov_tok = '<OOV>'
pad_type = 'post'
trunc_type = 'post'
kernal_size = 3
pool_size = 3
output_dim = 128
num_classes = 6
batch_size = 128
epoch = 20
Train_size = 20000
Validation_size = 4000
Test_size = 4000

data_path =  os.path.join(os.getcwd(), 'Data/train.csv')
preprocessed_path = os.path.join(os.getcwd(), 'Data/preprocessed_text.csv')
model_path =  os.path.join(os.getcwd(), 'Data/cnn_classifier.json')
model_weights =  os.path.join(os.getcwd(), 'Data/cnn_classifier.h5')