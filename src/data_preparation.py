"""Simple data preparation example for predictive maintenance dataset.
Reads data, performs basic cleaning and returns X, y.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    # Example features
    features = ['usage_hours', 'temp_c', 'pressure_bar', 'vibration_g', 'rpm']
    X = df[features].copy()
    y = df['failure'].copy()
    # Fill missing (if any) and scale
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

if __name__ == '__main__':
    X, y, scaler = load_and_prepare('../data/sample_data.csv')
    print('Prepared X shape:', X.shape)