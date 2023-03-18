# Script to test machine learning model performance in data slices
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load data.
data = pd.read_csv('./data/cleaned_data.csv')

# Load model artifacts
model = joblib.load('./model/model.pkl')
encoder = joblib.load('./model/encoder.pkl')
lb = joblib.load('./model/lb.pkl')

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

slice_metrics = []

def test_slice(data, feature):
    """ Function for testing model performance on slices of the Census dataset.
            - education slice"""
    for cls in data[feature].unique():
        df_temp = data[data[feature] == cls]
        
        X_test, y_test, _, _ = process_data(
            df_temp, 
            categorical_features=cat_features, 
            label="salary", 
            training=False, 
            encoder=encoder, 
            lb=lb
        )
        
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        line = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
        slice_metrics.append(line)

        with open('starter/slice_output.txt', 'w') as file:
            for line in slice_metrics:
                file.write(line + '\n')
   

if __name__ == '__main__':
    for feature in cat_features:
        test_slice(data, feature)
