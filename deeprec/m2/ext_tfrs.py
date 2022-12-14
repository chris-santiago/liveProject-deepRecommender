import pandas as pd
import tensorflow as tf

from deeprec import ROOT
from deeprec.m2.basic_tfrs import load_metadata, make_embedding_block, TowersRecommender

DATA_DIR = ROOT.joinpath('data')


def make_datasets():
    train = pd.read_parquet(DATA_DIR.joinpath('train.parq.gzip'))
    test = pd.read_parquet(DATA_DIR.joinpath('test.parq.gzip'))
    data = []
    for ds in [train, test]:
        dd = ds.iloc[:, :10].to_dict('list')
        dd.update({'genres': ds.iloc[:, 10:28].values})
        dd.update({'gender': ds.iloc[:, 28:30].values})
        dd.update({'age': ds.iloc[:, 30:37].values})
        dd.update({'occupation': ds.iloc[:, 37:-25].values})
        dd.update({'title_embeds': ds.iloc[:, -25:].values})
        data.append(tf.data.Dataset.from_tensor_slices(dd))
    return tuple(data)


class TowersModel(tf.keras.Model):
    def __init__(self, users, movies, cities, states, embed_dim=32):
        super().__init__()
        self.users = users
        self.movies = movies
        self.cities = cities
        self.states = states
        self.embed_dim = embed_dim

        self.months = [
            'November', 'August', 'December', 'July', 'May', 'June', 'September', 'October',
            'January', 'April', 'February', 'March'
        ]

        self.days_of_week = [
            'Monday', 'Tuesday', 'Sunday', 'Thursday', 'Wednesday', 'Friday', 'Saturday'
        ]
        self.movie_inputs = tf.keras.layers.Concatenate(axis=-1)

        # self.occupations = tf.keras.Input(shape=(1,), name='occupation_inputs')
        # self.hour = tf.keras.Input(shape=(1,), name='hour_inputs')
        # self.occupations = make_embedding_block(
        #     self.days_of_week, embed_dim=8, name='occupation_embeddings'
        # )
        self.user_embeds = make_embedding_block(
            self.users, embed_dim=self.embed_dim, name='user_embeddings'
        )
        # self.gender = make_embedding_block(
        #     ['M', 'F'], embed_dim=4, name='gender_embeddings'
        # )
        # self.day = make_embedding_block(
        #     self.days_of_week, embed_dim=4, name='day_embeddings'
        # )
        # self.month = make_embedding_block(
        #     self.months, embed_dim=4, name='month_embeddings'
        # )
        self.city_embeds = make_embedding_block(
            self.cities, embed_dim=self.embed_dim, name='city_embeddings'
        )
        self.state_embeds = make_embedding_block(
            self.states, embed_dim=self.embed_dim, name='state_embeddings'
        )
        self.user_inputs = tf.keras.layers.Concatenate(axis=-1)

        self.movie_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
            ],
            name='movie_model'
        )

        self.user_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
            ],
            name='movie_model'
        )

        self.all_inputs = tf.keras.layers.Concatenate(axis=-1)
        self.merge_model = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        # movie tower
        title = inputs['title_embeds']
        genre = tf.cast(inputs['genres'], dtype=tf.float64)
        year = tf.expand_dims(tf.cast(inputs['year'], dtype=tf.float64), axis=1)
        movie_inputs = self.movie_inputs([title, genre, year])
        movie_model = self.movie_model(movie_inputs)

        # user tower
        hour = tf.expand_dims(tf.cast(inputs['hour'], dtype=tf.float64), axis=1)
        occupation = tf.cast(inputs['occupation'], dtype=tf.float64)
        user = self.user_embeds(tf.strings.as_string(inputs['user']))
        gender = tf.cast(inputs['gender'], dtype=tf.float64)
        age = tf.cast(inputs['age'], dtype=tf.float64)
        day = tf.expand_dims(tf.cast(inputs['day_of_week'], dtype=tf.float64), axis=1)
        month = tf.expand_dims(tf.cast(inputs['month'], dtype=tf.float64), axis=1)
        city = self.city_embeds(inputs['city'])
        state = self.state_embeds(inputs['state'])
        user_inputs = self.user_inputs([
            occupation, hour, user, gender, age, day, month, city, state
        ])
        user_model = self.user_model(user_inputs)

        # merged model
        all_inputs = self.all_inputs([movie_model, user_model])
        return self.merge_model(all_inputs)


if __name__ == '__main__':
    tf_train, tf_test = make_datasets()
    meta = load_metadata()

    users = list(meta['user'].keys())
    movies = list(meta['movie'].keys())
    # occupations = list(meta['occupation'].keys())
    cities = list(meta['city'].keys())
    states = list(meta['state'].keys())

    model = TowersModel(users, movies, cities, states, embed_dim=16)
    rec = TowersRecommender(model)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=4096, decay_rate=0.95)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', restore_best_weights=True, patience=5
    )

    cached_train = tf_train.shuffle(10_000).batch(1024).cache()
    rec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    rec.fit(cached_train, epochs=100, callbacks=[early_stopping])

    cached_test = tf_test.batch(2048).cache()
    rec.evaluate(cached_test, return_dict=True)

    rec.model.save_weights(DATA_DIR.joinpath('ext_rec/'))
