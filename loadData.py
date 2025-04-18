import pandas as pd

df = pd.read_csv("Gujarat_Cost_of_Living_and_Sustainability.csv", index_col=0)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
