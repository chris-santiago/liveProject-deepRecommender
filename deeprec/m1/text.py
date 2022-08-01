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


def preprocess(data, model, col='title'):
    return pd.DataFrame(
        [embed_title(model, title) for title in clean_series(data[col])],
        columns=[f'embed_{i}' for i in range(model.vector_size)]
    )


if __name__ == '__main__':
    file = DATA_DIR.joinpath('dataset.parq.gzip')
    data = pd.read_parquet(file)

    model = api.load("glove-twitter-25")
    embeds = preprocess(data, model)
    embeds.to_parquet(DATA_DIR.joinpath('embeds.parq.gzip'), compression='gzip')

    data.drop('title', axis=1)\
        .join(embeds)\
        .to_parquet(DATA_DIR.joinpath('final.parq.gzip'), compression='gzip')
