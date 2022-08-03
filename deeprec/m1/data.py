import re

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from uszipcode import SearchEngine

from deeprec import ROOT


DATA_DIR = ROOT.joinpath('data')
ENCODING = "latin-1"


def load_data():
    data = {}
    for file in ['users', 'movies', 'ratings']:
        data[file] = pd.read_csv(DATA_DIR.joinpath(f'{file}.csv'), encoding=ENCODING)
    return data


def get_title(text):
    return text.strip()[:-7]
    # return ''.join(re.findall(r"\w+.+\s", text)).strip()


def get_year(text):
    return int(re.findall(r'(19\d{2}|20\d{2})', text)[-1])


def clean_genres(genres):
    return [re.sub(r'[^a-z0-9]+', '', s.lower()) for s in genres]


def split_genres(series):
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(series.str.split('|'))
    return pd.DataFrame(arr, columns=clean_genres(mlb.classes_))


def convert_timestamps(series, use_names=True):
    ts = pd.to_datetime(series, unit='s')
    if use_names:
        return pd.DataFrame(
            {'hour': ts.dt.hour, 'day_of_week': ts.dt.day_name(), 'month': ts.dt.month_name()})
    return pd.DataFrame({'hour': ts.dt.hour, 'day_of_week': ts.dt.dayofweek, 'month': ts.dt.month})


def std_zip(zipcode):
    return zipcode[:5]


def get_city_state(zipcode, engine):
    res = engine.by_zipcode(std_zip(zipcode))
    try:
        return res.major_city, res.state_abbr
    except AttributeError:
        return None, None


def expand_zips(series, engine):
    df = pd.DataFrame([get_city_state(x, engine) for x in series], columns=['city', 'state'])
    df['zip'] = series.apply(std_zip)
    return df


def preprocess(dd, use_names=True, na_val='XX'):
    search = SearchEngine()

    movies = pd.DataFrame({
        'movie': dd['movies']['movie'],
        'title': dd['movies']['title'].apply(get_title),
        'year': dd['movies']['title'].apply(get_year),
    }).join(split_genres(dd['movies']['genres']))

    ratings = dd['ratings'].iloc[:, :-1].join(convert_timestamps(dd['ratings']['timestamp'], use_names))
    users = dd['users'].iloc[:, :-1].join(expand_zips(dd['users']['zip'], search))

    return ratings.merge(users, on='user').merge(movies, on='movie').fillna(na_val)


if __name__ == '__main__':
    data = load_data()
    res = preprocess(data)
    res.to_parquet(DATA_DIR.joinpath('dataset.parq.gzip'), compression='gzip')
