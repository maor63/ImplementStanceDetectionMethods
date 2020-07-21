import re

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from scipy.spatial.distance import cosine
import gensim.downloader


def clean_text(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def preprocess_data(train_bodies_df, train_stance_df, word2index):
    bodies_clean_df = train_bodies_df.copy()
    ##### split to paragraphs ########
    bodies_clean_df = bodies_clean_df.assign(paragraph=bodies_clean_df['articleBody'].str.split('\n\n')).explode(
        'paragraph')

    ##### clean text ########
    bodies_clean_df['articleBody'] = bodies_clean_df['articleBody'].apply(clean_text)
    bodies_clean_df['paragraph'] = bodies_clean_df['paragraph'].apply(clean_text).str.split()
    bodies_clean_df = bodies_clean_df[bodies_clean_df['paragraph'].str.len() > 3]

    unknown_idx = -1
    # unknown_idx = word2index['<unknown>']
    space_idx = -2
    # space_idx = word2index['<space>']
    bodies_clean_df['paragraph'] = bodies_clean_df['paragraph'].apply(
        lambda words: [word2index.get(w, unknown_idx) for w in words])

    bodies_clean_df['paragraph'] = list(map(list, pad_sequences(bodies_clean_df['paragraph'], maxlen=15, padding='post',
                                                                value=space_idx)))

    body_to_paragraph_features_map = {}
    for body_id, group in bodies_clean_df.groupby('Body ID')['paragraph']:
        paragraphs = np.stack(group.iloc[:9].values, axis=0)
        par_count = len(paragraphs)
        paragraphs_pad = np.pad(paragraphs, ((0, 9 - par_count), (0, 0)), 'constant', constant_values=space_idx)
        body_to_paragraph_features_map[body_id] = paragraphs_pad

    train_stance_clean_df = train_stance_df.copy()
    train_stance_clean_df['Headline'] = train_stance_clean_df['Headline'].apply(clean_text)
    train_stance_clean_df['Headline'] = train_stance_clean_df['Headline'].apply(
        lambda words: [word2index.get(w, unknown_idx) for w in words])
    train_stance_clean_df['Headline'] = list(map(list, pad_sequences(train_stance_clean_df['Headline'], maxlen=15,
                                                                     padding='post', value=space_idx)))

    ##### mearge bodies and headline ########
    body_par_df = pd.DataFrame(body_to_paragraph_features_map.items(), columns=['Body ID', 'paragraph'])
    train_data = train_stance_clean_df.merge(body_par_df, on='Body ID')
    return train_data


def main():
    dataset_path = Path('data/FNC_data/')
    train_bodies_df = pd.read_csv(dataset_path / 'train_bodies.csv')
    train_stance_df = pd.read_csv(dataset_path / 'train_stances.csv')

    test_bodies_df = pd.read_csv(dataset_path / 'competition_test_bodies.csv')
    test_stance_df = pd.read_csv(dataset_path / 'competition_test_stances.csv')

    # model = gensim.downloader.load("glove-twitter-100")
    # word2index = {w: i for i, w in enumerate(model.index2word)}
    # embedding_weights = model.vectors
    all_words = [w for words in train_bodies_df['articleBody'].apply(clean_text).str.split().to_numpy() for w in words]
    word2index = {w: i for i, w in enumerate(all_words)}

    train_data = preprocess_data(train_bodies_df, train_stance_df, word2index)
    test_data = preprocess_data(test_bodies_df, test_stance_df, word2index)
    print('train test sizes')
    print(train_data.shape, test_data.shape)

    label_encoder = LabelEncoder()
    X_train, y_train, groups_train = train_data[['Headline', 'paragraph']], to_categorical(
        label_encoder.fit_transform(train_data['Stance'])), train_data['Body ID']
    X_test, y_test, groups_test = test_data[['Headline', 'paragraph']], to_categorical(
        label_encoder.transform(test_data['Stance'])), test_data['Body ID']
    print('class encoding')
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    print("Hello World!")


if __name__ == "__main__":
    main()
