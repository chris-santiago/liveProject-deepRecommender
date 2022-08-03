import json

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from deeprec import ROOT
from deeprec.m2.baselines import get_datasets

DATA_DIR = ROOT.joinpath('data')


def load_metadata():
    with open(DATA_DIR.joinpath('metadata.json'), 'r') as fp:
        return json.load(fp)


def make_datasets(cols=['user', 'movie', 'rating']):
    train = pd.read_parquet(DATA_DIR.joinpath('train.parq.gzip'))
    test = pd.read_parquet(DATA_DIR.joinpath('test.parq.gzip'))
    data = []
    for ds in [train, test]:
        data.append(
            tf.data.Dataset.from_tensor_slices(
                {k: list(v) for k, v in ds[cols].to_dict().items()}
            )
        )
    return tuple(data)


class TowersModel(tf.keras.Model):
    def __init__(self, users, movies, embed_dim=32):
        super().__init__()
        self.users = users
        self.movies = movies
        self.embed_dim = embed_dim

        self.user_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.StringLookup(vocabulary=users),
                tf.keras.layers.Embedding(len(users)+1, self.embed_dim),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
            ]
        )

        self.movie_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.StringLookup(vocabulary=movies),
                tf.keras.layers.Embedding(len(movies)+1, self.embed_dim),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
            ]
        )

        self.merge_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        user, movie, _ = inputs
        user_embeds = self.user_model(user)
        movie_embeds = self.movie_model(movie)
        return self.merge_model(tf.concat[user_embeds, movie_embeds], axis=1)


class TowersRecommender(tfrs.models.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):
        return self.model(inputs)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        return self.task(labels=inputs['rating'], predictions=self(inputs))


if __name__ == '__main__':
    cols = ['user', 'movie', 'rating']
    train, test = make_datasets(cols)
    meta = load_metadata()

    users = list(meta['user'].keys())
    movies = list(meta['movie'].keys())
    model = TowersModel(users, movies)
