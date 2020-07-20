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


def clean_text(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def preprocess_data(train_bodies_df, train_stance_df):
    bodies_clean_df = train_bodies_df.copy()
    ##### split to paragraphs ########
    bodies_clean_df = bodies_clean_df.assign(paragraph=bodies_clean_df['articleBody'].str.split('\n\n')).explode(
        'paragraph')

    ##### clean text ########
    bodies_clean_df['articleBody'] = bodies_clean_df['articleBody'].apply(clean_text)
    bodies_clean_df['paragraph'] = bodies_clean_df['paragraph'].apply(clean_text).str.split()
    bodies_clean_df = bodies_clean_df[bodies_clean_df['paragraph'].str.len() > 3]


    bodies_clean_df['paragraph'] = pad_sequences(bodies_clean_df['paragraph'], maxlen=15, padding='post', value=0)

    for body_id, group in bodies_clean_df.groupby('Body ID')[['paragraph']]:
        paragraphs = group.reset_index().iloc[:9]

    train_stance_clean_df = train_stance_df.copy()
    train_stance_clean_df['Headline'] = train_stance_clean_df['Headline'].apply(clean_text)

    ##### mearge bodies and headline ########
    train_data = train_stance_clean_df.merge(bodies_clean_df, on='Body ID')
    return train_data


def main():
    dataset_path = Path('data/FNC_data/')
    train_bodies_df = pd.read_csv(dataset_path / 'train_bodies.csv')
    train_stance_df = pd.read_csv(dataset_path / 'train_stances.csv')

    test_bodies_df = pd.read_csv(dataset_path / 'competition_test_bodies.csv')
    test_stance_df = pd.read_csv(dataset_path / 'competition_test_stances.csv')

    train_data = preprocess_data(train_bodies_df, train_stance_df)
    test_data = preprocess_data(test_bodies_df, test_stance_df)
    print('train test sizes')
    print(train_data.shape, test_data.shape)
    all_words = list(set(train_data['']))


    label_encoder = LabelEncoder()
    X_train, y_train, groups_train = train_data[['Headline', 'articleBody']], to_categorical(
        label_encoder.fit_transform(train_data['Stance'])), train_data['Body ID']
    X_test, y_test, groups_test = test_data[['Headline', 'articleBody']], to_categorical(
        label_encoder.transform(test_data['Stance'])), test_data['Body ID']
    print('class encoding')
    print(dict(*zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    print("Hello World!")


if __name__ == "__main__":
    main()
