from pathlib import Path
import logging
import pandas as pd
import pytest
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import yaml

with open("../config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

cat_features = config['cat_features']

@pytest.fixture(name='data')
def data():
    """
    Fixture will be used by the unit tests.
    """
    yield pd.read_csv('../data/census_cleaned.csv')


def test_raw_data(data):
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


def test_model():
    model = joblib.load('../model/model.pkl')
    assert isinstance(model, RandomForestClassifier)


def test_processed_data(data):
    X_train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(X_train, cat_features, label='salary')
    assert len(X) == len(y)
    assert len(X_train) == len(X)