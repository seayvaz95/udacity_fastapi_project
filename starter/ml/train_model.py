# Script to train machine learning model.
import logging
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference
import pandas as pd
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info("Loading data")
data = pd.read_csv("../data/census_cleaned.csv")

with open("../config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info("Splitting data")
train, test = train_test_split(data, test_size=0.20)

cat_features = config['cat_features']

logging.info("Processing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
logging.info("Training the model")
model = train_model(X_train, y_train, config['random_forest'])

logging.info("Predicting on test data")
y_pred = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}, recall: {recall: .2f}, fbeta: {fbeta: .2f}")

joblib.dump(model, '../model/model.pkl')
joblib.dump(encoder, '../model/encoder.pkl')
joblib.dump(lb, '../model/lb.pkl')