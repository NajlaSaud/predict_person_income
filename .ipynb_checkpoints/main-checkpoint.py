from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Literal
import numpy as np
import pandas as pd
import joblib
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

# Input data
class InputData(BaseModel):
    age: int
    workclass: Literal['State-gov', 
                       'Self-emp-not-inc', 
                       'Private', 
                       'Federal-gov', 
                       'Local-gov', 
                       'Self-emp-inc', 
                       'Without-pay', 
                       'Never-worked']
    fnlgt: int 
    education: Literal['Bachelors', 
                       'HS-grad', 
                       '11th', 
                       'Masters', 
                       '9th', 
                       'Some-college',
                       'Assoc-acdm', 
                       'Assoc-voc', 
                       '7th-8th', 
                       'Doctorate', 
                       'Prof-school',
                       '5th-6th', 
                       '10th', 
                       'Preschool', 
                       '12th', 
                       '1st-4th']
    education_num: int 
    marital_status: Literal['Never-married', 
                            'Married-civ-spouse', 
                            'Divorced',
                            'Married-spouse-absent', 
                            'Separated', 
                            'Married-AF-spouse',
                            'Widowed']
    occupation: Literal['Adm-clerical', 
                        'Exec-managerial', 
                        'Handlers-cleaners', 
                        'Prof-specialty', 
                        'Other-service', 
                        'Sales', 
                        'Craft-repair',
                        'Transport-moving', 
                        'Farming-fishing', 
                        'Machine-op-inspct',
                        'Tech-support', 
                        'Protective-serv', 
                        'Armed-Forces',
                        'Priv-house-serv']
    relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
       'Other-relative']
    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
       'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico',
       'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
       'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
       'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
       'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
       'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
       'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
       'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
       'Holand-Netherlands']
    
class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": 'Private',
                "fnlgt": 88416,
                "education": 'Masters',
                "education_num": 13,
                "marital_status": "Married-spouse-absent",
                "occupation": "Tech-support",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 35,
                "native_country": 'United-States'
            }
        }

# Load model artifacts
model = joblib.load('./model/model.pkl')
encoder = joblib.load('./model/encoder.pkl')
lb = joblib.load('./model/lb.pkl')
    


# GET on the root giving a welcome message.
@app.get("/")
async def say_hello():
    return {
        "greeting": "Hello! I`m Najla. This is my project on machine learning pipelines.",
        "description": "This model predicts whether income exceeds $50K/yr based on census data."
    }

# POST on predict-income taking data and giving inference
@app.post("/predict-income")
async def predict(input: InputData):
    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    input_df = pd.DataFrame(data=input_data, columns=columns)
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

    X, _, _, _ = process_data(
        input_df, 
        categorical_features=cat_features, 
        label="salary",
        encoder=encoder, 
        lb=lb, 
        training=False)
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)