import gensim.downloader
import tensorflow as tf
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def Conv2DWithMaxPool2D(x, filter=3, embedding_size=300):
    filters = Conv2D(100, filter, activation='relu', kernel_regularizer=1e-6)(x)
    filters = MaxPooling2D(pool_size=(sentence_max_len - filter + 1, embedding_size - filter + 1))(filters)
    return filters

def create_model(sentence_max_len ,index2word, embedding_matrix):
    input = Input(shape=(sentence_max_len,), dtype='int32', name='input_vector')
    x = Embedding(len(index2word) + 1, 300, weights=[embedding_matrix], input_length=sentence_max_len, trainable=False)(
        input)
    x = tf.keras.backend.expand_dims(x, axis=-1)
    filter3 = Conv2DWithMaxPool2D(x, 3, 300)
    filter4 = Conv2DWithMaxPool2D(x, 4, 300)
    filter5 = Conv2DWithMaxPool2D(x, 5, 300)

    x = concatenate([filter3, filter4, filter5])
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax', kernel_regularizer=1e-6)(x)

    model = Model(input=input, output=output)
    return model

#
df = pd.read_csv('data/semeval2016-task6-trainingdata.csv', sep='\t', encoding='ascii')

wv = gensim.downloader.load('word2vec-google-news-300')
index2word = wv.index2word
embedding_matrix = wv.syn0

texts = df['Tweet']
sentence_max_len = texts.str.split().apply(len).max()

word2index = dict(zip(index2word, range(len(index2word))))

encoded_corpus = texts.str.split().apply(lambda xs: list(map(word2index.get, xs)))

encoded_corpus_pad = pad_sequences(encoded_corpus, maxlen=sentence_max_len, padding='post', value=0)

