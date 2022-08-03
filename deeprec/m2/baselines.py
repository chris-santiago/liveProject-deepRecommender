import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error

from deeprec import ROOT

DATA_DIR = ROOT.joinpath('data')


class BaseContainer(BaseEstimator, RegressorMixin):
    def __init__(self):
        super().__init__()

    def _fit(self, X, y):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self._fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        return self._predict(X)


class RandomGuessEstimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        _ = X
        self.ratings_ = set(y)

    def predict(self, X):
        rng = np.random.default_rng()
        return rng.integers(low=min(self.ratings_), high=max(self.ratings_), size=len(X))


class WeightedSamplingEstimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        _ = X
        self.weights_ = y.value_counts(normalize=True, ascending=True).to_dict()

    def predict(self, X):
        rng = np.random.default_rng()
        return rng.choice(a=list(self.weights_.keys()), size=len(X), p=list(self.weights_.values()))


class MajorityClassEstimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        _ = X
        self.mode_ = y.mode()

    def predict(self, X):
        return np.repeat(self.mode_, repeats=len(X))


class MeanValueEstimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        _ = X
        self.mean_ = y.mean()

    def predict(self, X):
        return np.repeat(self.mean_, repeats=len(X))


def get_datasets():
    train = pd.read_parquet(DATA_DIR.joinpath('train.parq.gzip'))
    test = pd.read_parquet(DATA_DIR.joinpath('test.parq.gzip'))
    X_cols = [c for c in train.columns if c != 'rating']

    X_train, y_train = train[X_cols], train['rating']
    X_test, y_test = test[X_cols], test['rating']
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_datasets()

    for est in [
        RandomGuessEstimator(), WeightedSamplingEstimator(),
        MajorityClassEstimator(), MeanValueEstimator()
    ]:
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f'Estimator: {est.__repr__()}, RMSE: {rmse}')
