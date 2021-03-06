# -*- coding: utf-8 -*-
"""Emotion_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IQW4YrhkiR-PfGujkKOspfe7oeKXyONK
"""

import pandas as pd
import numpy as np
import scipy
import sklearn
import eval as meu
import nltk

train_data = pd.read_csv('/content/drive/My Drive/Emotion-Detection/normailized_data/imdb_train.csv')
test_data = pd.read_csv('/content/drive/My Drive/Emotion-Detection/normailized_data/imdb_test.csv')

train_data.shape
train_data.head()

test_data.shape

"""Machine learning models and evaluation"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1,2))
cv_train_features = cv.fit_transform(train_data["text"])
cv_test_features = cv.transform(test_data["text"])
cv_train_labels = train_data["polarity"]
cv_test_labels = test_data["polarity"]

# build Logistic Regression model
lr = LogisticRegression(penalty='l2', max_iter=100, C=1)

lr_bow_predictions = meu.train_predict_model(classifier=lr,train_features=cv_train_features, train_labels=cv_train_labels,
                                             test_features=cv_test_features, test_labels=cv_test_labels)
meu.display_model_performance_metrics(true_labels=cv_test_labels, predicted_labels=lr_bow_predictions)

import keras
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Conv1D,Flatten
from keras.layers import Embedding

"""Prediction class label encoding
For the deep learning models we use the one-hot encoding to change the sentiment labels to numeric representations.
"""

le = LabelEncoder()
num_classes=10 
# encode train labels
y_tr = le.fit_transform(train_data["polarity"])
y_train = keras.utils.to_categorical(y_tr, num_classes)
# encode test labels
y_ts = le.fit_transform(test_data["polarity"])
y_test = keras.utils.to_categorical(y_ts, num_classes)

print('Sentiment class label map:', dict(zip(le.classes_, le.transform(le.classes_))))
print('Sample test label transformation:\n'+'-'*35,
      '\nActual Labels:', test_data["polarity"][:3], '\nEncoded Labels:', y_ts[:3], 
      '\nOne hot encoded Labels:\n', y_test[:3])

import gensim
w2v_num_features = 500
w2v_model = gensim.models.Word2Vec(train_data["text"], size=w2v_num_features, window=150,
                                   min_count=10, sample=1e-3)

# Function to compute averaged word vector representations for corpus of text documents
def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)

# generate averaged word vector features from word2vec model
avg_wv_train_features = averaged_word2vec_vectorizer(corpus=train_data["text"], model=w2v_model,
                                                     num_features=500)
avg_wv_test_features = averaged_word2vec_vectorizer(corpus=test_data["text"], model=w2v_model,
                                                    num_features=500)
print(avg_wv_train_features)

print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape, ' Test features shape:', avg_wv_test_features.shape)
print('train',avg_wv_train_features)
print('test',avg_wv_test_features)
print('label',y_train)

from keras.layers import Embedding

class Dnn:
    def __init__(self):
        pass


    def construct_dnn(self,num_input_features=500):
        dnn_model = Sequential()
        dnn_model.add(Dense(512, activation='relu', input_shape=(num_input_features,)))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(512, activation='relu'))
        dnn_model.add(Dropout(0.2))
        dnn_model.add(Dense(10))
        dnn_model.add(Activation('softmax'))

        dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                          metrics=['accuracy'])
        dnn_model.summary()
        return dnn_model 


    def construct_cnn(self,embedding_len=32,total_vocab=5000,upper_threshold=256):
        model = Sequential()
        model.add(Embedding(total_vocab,embedding_len,input_length = upper_threshold))
        model.add(Conv1D(128,3,padding = 'same'))
        model.add(Conv1D(64,3,padding = 'same'))
        model.add(Conv1D(32,2,padding = 'same'))
        model.add(Conv1D(16,2,padding = 'same'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(100,activation = 'sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation='sigmoid'))
        model.summary()
        return model

model_class = Dnn()
dnn = model_class.construct_cnn()

batch_size = 100
dnn.fit(avg_wv_train_features, y_train, epochs=5, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)

y_pred = dnn.predict_classes(avg_wv_test_features)
predictions = le.inverse_transform(y_pred)
print(predictions)

meu.display_model_performance_metrics(true_labels=test_data['polarity'], predicted_labels=predictions, classes=[1, 2, 3, 4, 7, 8, 9, 10])

