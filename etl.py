
import pandas as pd

def load_data(path='file:///C:/Users/User/Downloads/ny-flights.csv'):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.dropna(subset=['Distance', 'ScheduledTime', 'Delayed'])
    df['Delayed'] = df['Delayed'].astype(int)
    return df

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print("Data loaded and cleaned:", df.shape)
