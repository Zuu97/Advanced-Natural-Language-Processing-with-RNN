import os
vocab_size = 20000
embedding_dim = 100
cutoff = 0.2
batch_size = 64
epochs = 50
d1 = 10
d2 = 1
dense_context = 100
seed = 42
hidden_dim_encoder = 400
hidden_dim_decoder = 400
num_samples = 10000
padding_type = 'post'
truncating_type = 'post'
data_dir = 'E:\My projects 2\Advanced-Natural-Language-Processing-with-RNN\Data'
text_path = os.path.join(data_dir,"spa.txt")

attention_model_path = os.path.join(data_dir,"attention_model.json")
attention_model_weights = os.path.join(data_dir,"attention_model.h5")
