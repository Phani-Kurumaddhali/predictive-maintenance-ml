"""Generate synthetic predictive maintenance dataset.

Usage:
    python generate_sample_data.py --output ../data/sample_data.csv --n 500
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate(n=500, seed=42):
    np.random.seed(seed)
    start = datetime(2024,1,1)
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    usage_hours = np.cumsum(np.abs(np.random.normal(loc=1.5, scale=0.5, size=n))).astype(int)
    temp = np.random.normal(loc=60, scale=3, size=n) + (usage_hours * 0.002)
    pressure = np.random.normal(loc=30, scale=1.5, size=n)
    vibration = np.abs(np.random.normal(loc=0.5, scale=0.2, size=n)) + (usage_hours * 0.0001)
    rpm = np.random.normal(loc=1500, scale=80, size=n)
    prob = 1 / (1 + np.exp(-((temp - 65)*0.35 + (vibration - 0.6)*3 + (usage_hours - 300)*0.001)))
    failure = (np.random.rand(n) < prob * 0.2).astype(int)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "usage_hours": usage_hours,
        "temp_c": np.round(temp, 2),
        "pressure_bar": np.round(pressure, 2),
        "vibration_g": np.round(vibration, 3),
        "rpm": np.round(rpm, 0).astype(int),
        "failure": failure
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='../data/sample_data.csv')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = generate(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Saved sample data to {args.output}")