from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import pytest
from starter.ml.data import process_data

data_path = './data/cleaned_data.csv'
model_path = './model/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    """
    Fixture for data to be used by other tests.
    """
    yield pd.read_csv(data_path)


def test_load_data(data):
    
    """ Check the data received is a pandas DataFrame and not empty """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


def test_model():

    """ Check model type is LogisticRegression """

    model = joblib.load(model_path)
    assert isinstance(model, LogisticRegression)


def test_process_data(data):

    """ Test the data split """

    train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)
