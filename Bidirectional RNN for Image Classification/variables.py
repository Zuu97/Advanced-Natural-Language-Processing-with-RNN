import os
image_size = 28
num_classes = 10
batch_size = 64
num_epochs = 10
lstm_dim = 15

train_pickle = os.path.join(os.getcwd(), 'Data/train')
test_pickle = os.path.join(os.getcwd(), 'Data/test')
train_path = os.path.join(os.getcwd(), 'Data/mnist_train.csv')
test_path = os.path.join(os.getcwd(), 'Data/mnist_test.csv')
model_path =  os.path.join(os.getcwd(), 'Data/mnist_classifier.json')
model_weights =  os.path.join(os.getcwd(), 'Data/mnist_classifier.h5')