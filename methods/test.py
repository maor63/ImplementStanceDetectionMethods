import gensim.downloader
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np


#
df = pd.read_csv('data/semeval2016-task6-trainingdata.csv', sep='\t', encoding='ascii')

wv = api.load('word2vec-google-news-300')
index2word = wv.index2word
embedding_matrix = wv.syn0

texts = df['Tweet']
sentence_max_len = texts.str.split().apply(len).max()
print(sentence_max_len)
print(wv['king'])