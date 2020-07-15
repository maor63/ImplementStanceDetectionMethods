import re
import tensorflow.keras.backend as K
import gensim.downloader
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import expand_dims
from tensorflow.keras import regularizers


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """
    print(ytrue.shape, ypreds.shape)

    y = K.argmax(ytrue, axis=1)

    log_likelihood = K.log(ypreds[:, y])

    return K.mean(-log_likelihood)


def Conv2DWithMaxPool2D(x, filter=3, embedding_size=300):
    filters = tf.keras.layers.Conv2D(100, filter, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(x)
    filters = tf.keras.layers.GlobalMaxPool2D()(filters)
    # filters = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(filters)
    # filters = tf.keras.layers.Flatten()(filters)
    return filters


def create_model(sentence_max_len, vocab_size, embedding_matrix):
    input = tf.keras.layers.Input(shape=(sentence_max_len,), dtype='int32', name='input_vector')
    x = tf.keras.layers.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=sentence_max_len, trainable=True)(input)
    # x = tf.keras.layers.Embedding(vocab_size, 300, input_length=sentence_max_len, trainable=True)(input)
    x = expand_dims(x, axis=-1)
    filter3 = Conv2DWithMaxPool2D(x, 3, 300)
    filter4 = Conv2DWithMaxPool2D(x, 4, 300)
    filter5 = Conv2DWithMaxPool2D(x, 5, 300)

    x = tf.keras.layers.concatenate([filter3, filter4, filter5])

    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(3, activation='softmax', name='output_vector')(x)

    model = tf.keras.Model(inputs=[input], outputs=[output])
    return model

def convet_to_word_indexs(splited_texts):
    encoded_corpus = splited_texts.apply(lambda xs: [word2index.get(x, vocab_size - 1) for x in xs])
    encoded_corpus_pad = pad_sequences(encoded_corpus, maxlen=sentence_max_len, padding='post', value=0)
    return encoded_corpus_pad

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()
#
df = pd.read_csv('data/semeval2016-task6-trainingdata.csv', sep='\t', encoding='ascii')
test_df = pd.read_csv('data/SemEval2016-Task6-subtaskA-testdata-gold.txt', sep='\t', encoding='ascii')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(df['Stance'].value_counts())

wv = gensim.downloader.load('word2vec-google-news-300')
index2word = np.copy(wv.index2word)
embedding_matrix = np.copy(wv.vectors)
del wv

splited_texts = df['Tweet'].apply(clean_str).str.split()
sentence_max_len = splited_texts.apply(len).max()

word2index = dict(zip(index2word, range(len(index2word))))
vocab_size = len(index2word)

encoded_corpus_pad = convet_to_word_indexs(splited_texts)

model = create_model(sentence_max_len, vocab_size, embedding_matrix)
print(model.summary())
# exit(0)

label_encoder = LabelEncoder()
y_true = to_categorical(label_encoder.fit_transform(df['Stance']))
print(label_encoder.classes_)
print(label_encoder.transform(label_encoder.classes_))

# opt = tf.keras.optimizers.Adam()
opt = tf.keras.optimizers.Adadelta(
    learning_rate=0.5, rho=0.95, epsilon=1e-06, name="Adadelta",
)
model.compile(loss={'output_vector': gaussian_nll}, optimizer=opt, metrics=['accuracy'])
model.fit(encoded_corpus_pad, y_true, epochs=25, batch_size=50, verbose=2, validation_split=0.1)
pass

test_encoded_corpus_pad = convet_to_word_indexs(test_df['Tweet'].apply(clean_str).str.split())
y_test = label_encoder.transform(test_df['Stance'])

y_proba = model.predict(test_encoded_corpus_pad)
y_pred = y_proba.argmax(axis=1)
print(classification_report(y_test, y_pred))