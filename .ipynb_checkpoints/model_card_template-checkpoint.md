# Model Card
Information about the machine learning model used in this project.


## Model Details

This model is created by Najla Alsaedi. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.

## Intended Use

This model should be used to predict whether income exceeds $50K/yr based on census data.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Census+Income). 

The original dataset has 32561 rows and 15 columns. After dropping the duplicate rows, it has 32537 rows. The data was breaked into train and test set with 80-20 split. 

To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics

The model was evaluated using:
- Precision: The value is  0.7445
- Recall: The recall is  0.2662
- F beta score: The value is 0.3921

