import os
max_length = 100
vocab_size = 2000
embedding_dim = 80
cutoff = 0.2
batch_size = 128
epochs = 100
hidden_dim = 25
oov = '<OOV>'
padding_type = 'post'
truncating_type = 'post'
data_dir = 'E:\My projects 2\Advanced-Natural-Language-Processing-with-RNN\Data'
text_path = os.path.join(data_dir,"robert_frost.txt")

encoder_path = os.path.join(data_dir,"encoder.json")
encoder_weights = os.path.join(data_dir,"encoder.h5")