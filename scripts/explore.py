# ===========================================
# 01_explore_data.py
# Purpose: Understand the datasets
# ===========================================

import pandas as pd

print("=" * 60)
print("LOADING DATASETS...")
print("=" * 60)

# Load Netflix dataset
netflix = pd.read_csv("data/raw/netflix_titles.csv")

# Load IMDb Basics (only required columns)
basics = pd.read_csv(
    "data/raw/title.basics.tsv",
    sep="\t",
    usecols=[
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres"
    ],
    low_memory=False
)

# Load IMDb Ratings
ratings = pd.read_csv(
    "data/raw/title.ratings.tsv",
    sep="\t"
)

print("\nDatasets Loaded Successfully!")

# ===========================================
# Netflix Dataset
# ===========================================

print("\n" + "="*60)
print("NETFLIX DATASET")
print("="*60)

print("\nFirst 5 Rows")
print(netflix.head())

print("\nLast 5 Rows")
print(netflix.tail())

print("\nShape")
print(netflix.shape)

print("\nColumns")
print(netflix.columns.tolist())

print("\nInformation")
print(netflix.info())

print("\nMissing Values")
print(netflix.isnull().sum())

print("\nDuplicate Rows")
print(netflix.duplicated().sum())

print("\nStatistics")
print(netflix.describe(include='all'))

# ===========================================
# IMDb Basics
# ===========================================

print("\n" + "="*60)
print("IMDB BASICS DATASET")
print("="*60)

print("\nFirst 5 Rows")
print(basics.head())

print("\nShape")
print(basics.shape)

print("\nColumns")
print(basics.columns.tolist())

print("\nInformation")
print(basics.info())

print("\nMissing Values")
print(basics.isnull().sum())

print("\nDuplicate Rows")
print(basics.duplicated().sum())

# ===========================================
# IMDb Ratings
# ===========================================

print("\n" + "="*60)
print("IMDB RATINGS DATASET")
print("="*60)

print("\nFirst 5 Rows")
print(ratings.head())

print("\nShape")
print(ratings.shape)

print("\nColumns")
print(ratings.columns.tolist())

print("\nInformation")
print(ratings.info())

print("\nMissing Values")
print(ratings.isnull().sum())

print("\nDuplicate Rows")
print(ratings.duplicated().sum())

print("\nStatistics")
print(ratings.describe())

print("\nData Exploration Completed Successfully!")