import os
vocab_size = 20000
embedding_dim = 100
cutoff = 0.2
batch_size = 128
epochs = 50
hidden_dim = 256
num_samples = 10000
padding_type = 'post'
truncating_type = 'post'
data_dir = 'E:\My projects 2\Advanced-Natural-Language-Processing-with-RNN\Data'
text_path = os.path.join(data_dir,"spa.txt")

teacher_forcing_model_path = os.path.join(data_dir,"teacher_forcing_model.json")
teacher_forcing_model_weights = os.path.join(data_dir,"teacher_forcing_model.h5")
