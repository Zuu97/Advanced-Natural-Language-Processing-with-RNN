import re
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from variables import data_path, preprocessed_path, Train_size, Test_size, Validation_size
def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocessed_text_column(row):
    review = str(row['comment_text'])
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return str(updated_review)

def get_data():
    if not os.path.exists(preprocessed_path):
        print("Preprocessing Data !!!")
        df = pd.read_csv(data_path)
        data = df.dropna(axis = 0, how ='any')
        data['preprocessed_text'] = data.apply(preprocessed_text_column, axis=1)
        data.to_csv(preprocessed_path, encoding='utf-8', index=False)

    data = pd.read_csv(preprocessed_path)
    data = shuffle(data)

    labels = data.iloc[:,2:-1].values
    texts = data['preprocessed_text'].values.astype(str)

    test_total = Train_size + Test_size
    val_total  = test_total + Validation_size
    Ytrain, Ytest, Yval = labels[:Train_size,], labels[Train_size: test_total,], labels[test_total: val_total,]
    Xtrain, Xtest, Xval = texts[:Train_size,] , texts[Train_size: test_total,] , texts[test_total: val_total,]

    print("Data is ready !!!")
    return Ytrain, Xtrain, Ytest, Xtest, Yval, Xval