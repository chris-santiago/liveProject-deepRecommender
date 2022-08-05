import json
from collections import defaultdict

import pandas as pd
import numpy as np

import gensim.downloader as api
from gensim.parsing import preprocess_string, strip_non_alphanum, strip_punctuation, strip_multiple_whitespaces

from deeprec import ROOT

DATA_DIR = ROOT.joinpath('data')


def clean_series(series):
    filters = [strip_non_alphanum, strip_punctuation, strip_multiple_whitespaces]
    return (preprocess_string(title.lower(), filters=filters) for title in series)


def safe_embed(model, token):
    try:
        return model[token]
    except KeyError:
        return np.zeros(model.vector_size)


def embed_title(model, title):
    if len(title) == 1:
        return safe_embed(model, title).flatten()
    embeds = np.array([safe_embed(model, t) for t in title])
    if embeds.ndim != 2:
        raise ValueError(f'Expected array with 2 dims, got {embeds.ndim}')
    return embeds.sum(0)


def get_embeds(data, model, col='title'):
    return pd.DataFrame(
        [embed_title(model, title) for title in clean_series(data[col])],
        columns=[f'embed_{i}' for i in range(model.vector_size)]
    )


def preprocess(data, model, write_embeds=False, dummies=None):
    embeds = get_embeds(data, model)
    if write_embeds:
        embeds.to_parquet(DATA_DIR.joinpath('embeds.parq.gzip'), compression='gzip')
    if dummies:
        data = pd.get_dummies(data, columns=dummies)
    return data.drop('title', axis=1).join(embeds)


def make_train_test(data, write=False):
    df = data.sort_values(by=['year', 'month', 'hour'])
    mask = df['year'] > 1999
    train, test = df[~mask], df[mask]
    if write:
        train.to_parquet(DATA_DIR.joinpath('train.parq.gzip'), compression='gzip')
        test.to_parquet(DATA_DIR.joinpath('test.parq.gzip'), compression='gzip')
    else:
        return train, test


def make_metadata(data, write=False):
    metadata = defaultdict(dict)
    cats = ['user', 'movie', 'day_of_week', 'month', 'gender', 'age', 'occupation', 'city', 'state',
            'zip', 'year']
    for cat in cats:
        if data[cat].nunique() > 25:
            res = data[cat].value_counts(normalize=True)
            sample = 0
            for k, v in res.to_dict().items():
                if sample > 0.8:
                    break
                # print(f'working on cat: {k}, {v}%')
                metadata[cat].update({k: v})
                sample += v
                # print(f'percent covered: {sample}')
        else:
            metadata[cat] = data[cat].value_counts(normalize=True).to_dict()
    if write:
        with open(DATA_DIR.joinpath('metadata.json'), 'w') as fp:
            json.dump(metadata, fp)
    else:
        return metadata


if __name__ == '__main__':
    file = DATA_DIR.joinpath('dataset.parq.gzip')
    data = pd.read_parquet(file)

    model = api.load("glove-twitter-25")
    final = preprocess(data, model)
    final.to_parquet(DATA_DIR.joinpath('final.parq.gzip'), compression='gzip')

    make_train_test(final, write=True)
    make_metadata(final, write=True)
