"""Simple predict script that loads the saved model and runs prediction on a CSV of new samples."""
import joblib
import pandas as pd
import numpy as np

def predict(input_csv, model_path='model/rf_model.joblib'):
    data = pd.read_csv(input_csv)
    model_pack = joblib.load(model_path)
    clf = model_pack['model']
    scaler = model_pack['scaler']
    features = ['usage_hours', 'temp_c', 'pressure_bar', 'vibration_g', 'rpm']
    X = data[features].copy()
    X.fillna(X.median(), inplace=True)
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)
    data['pred_failure'] = preds
    return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/sample_data.csv')
    parser.add_argument('--model', default='model/rf_model.joblib')
    args = parser.parse_args()
    out = predict(args.input, args.model)
    print(out[['timestamp','failure','pred_failure']].head())