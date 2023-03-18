# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load data.
data = pd.read_csv('./data/cleaned_data.csv')


train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print('precision %: ', precision)
print('recall %: ', recall)
print('fbeta %: ', fbeta)

pickle.dump(model, open('./model/model.pkl','wb'))
pickle.dump(encoder, open('./model/encoder.pkl','wb'))
pickle.dump(lb, open('./model/lb.pkl','wb'))

