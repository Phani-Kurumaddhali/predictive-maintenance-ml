"""Train a RandomForest model on the sample dataset and save it.
"""
import argparse
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.data_preparation import load_and_prepare

def train(input_path, model_out='model/rf_model.joblib'):
    X, y, scaler = load_and_prepare(input_path)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, model_out)
    preds = clf.predict(X)
    print('Model trained. Classification report:')
    print(classification_report(y, preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/sample_data.csv')
    parser.add_argument('--model_out', default='model/rf_model.joblib')
    args = parser.parse_args()
    train(args.input, args.model_out)