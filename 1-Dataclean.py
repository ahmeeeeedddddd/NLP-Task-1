from PIL import GimpGradientFile
import pandas as pd
import numpy as np

data = pd.read_csv("imdb_messy.csv")

print(data.shape)
print(data.head())
data.info()

print(60*"/")

print(data.isnull().sum())

print("Total rows with a null value: ", data.isnull().any(axis=1).sum())#axis=1: will go row by row

before = len(data)
print("Before dropping null values: ",before)
data = data.dropna()
print(f"Dropped {before - len(data)} rows with a null value, Remaining: {len(data)}")

before = len(data)
print("Duplicated rows:", data.duplicated().sum())
data = data.drop_duplicates()
data = data.reset_index(drop=True)
print(f"Dropped {before - len(data)} duplicates, Remaining: {len(data)}")

data.to_csv("imdb_cleaned.csv", index=False)
