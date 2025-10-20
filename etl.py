import pandas as pd
import os

def load_data(path='ny-flights.csv'):
    """
    Load flight data from a CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """
    Clean the dataset:
    - Drop rows with missing crucial values
    - Ensure numeric columns are correctly typed
    """
    df = df.dropna(subset=['arr_delay', 'distance', 'sched_arr_time', 'month', 'day', 'carrier'])
    # Convert numeric columns
    numeric_cols = ['arr_delay', 'distance', 'sched_arr_time', 'month', 'day']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)
    return df
