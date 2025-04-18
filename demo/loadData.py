# module1_load_data.py

import pandas as pd

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(" Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(" File not found. Please check the path.")
    except Exception as e:
        print(" An error occurred:", e)

def inspect_data(df):
    print("\n First 5 rows of the dataset:")
    print(df.head())

    print("\n Dataset Info:")
    print(df.info())

    print("\n Missing Values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    df = load_data("gujarat_data.csv")
    if df is not None:
        inspect_data(df)
