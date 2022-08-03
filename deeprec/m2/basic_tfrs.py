import tensorflow as tf
import tensorflow_recommenders as tfrs

from deeprec import ROOT
from deeprec.m2.baselines import get_datasets

DATA_DIR = ROOT.joinpath('data')


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_datasets()
