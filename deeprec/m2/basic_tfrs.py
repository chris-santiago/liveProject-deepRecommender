import json

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from deeprec import ROOT

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
                ds[cols].to_dict('list')
            )
        )
    return tuple(data)


def make_embedding_block(vocab, embed_dim=32, name=None):
    return tf.keras.Sequential(
        layers=[
            tf.keras.layers.StringLookup(vocabulary=vocab),
            tf.keras.layers.Embedding(len(vocab) + 1, embed_dim),
        ],
        name=name
    )


class TowersModel(tf.keras.Model):
    def __init__(self, users, movies, embed_dim=32):
        super().__init__()
        self.users = users
        self.movies = movies
        self.embed_dim = embed_dim

        self.user_embeds = make_embedding_block(
            self.users, embed_dim=self.embed_dim, name='user_embeddings'
        )
        self.movie_embeds = make_embedding_block(
            self.movies, embed_dim=self.embed_dim, name='movie_embeddings'
        )
        self.merge_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        user = tf.strings.as_string(inputs['user'])
        movie = tf.strings.as_string(inputs['movie'])
        user_embeds = self.user_embeds(user)
        movie_embeds = self.movie_embeds(movie)
        return self.merge_model(tf.concat([user_embeds, movie_embeds], axis=-1))


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
    tf_train, tf_test = make_datasets(cols)
    meta = load_metadata()

    users = list(meta['user'].keys())
    movies = list(meta['movie'].keys())
    model = TowersModel(users, movies, embed_dim=32)
    rec = TowersRecommender(model)

    learning_rate = 1e-3
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', restore_best_weights=True, patience=5
    )

    cached_train = tf_train.shuffle(10_000).batch(2048).cache()
    rec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    rec.fit(cached_train, epochs=100, callbacks=[early_stopping])

    cached_test = tf_test.batch(4096).cache()
    rec.evaluate(cached_test, return_dict=True)

    rec.model.save_weights(DATA_DIR.joinpath('base_rec/'))
