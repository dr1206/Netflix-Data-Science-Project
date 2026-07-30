import pandas as pd

merged = pd.read_csv("data/cleaned/merged_v1.csv")

print("Shape:", merged.shape)

print("\nMissing values in IMDb columns:")
print(merged[["averageRating", "numVotes"]].isnull().sum())

print("\nFirst 10 merged rows:")
print(
    merged[
        ["title", "release_year", "primaryTitle", "startYear", "averageRating"]
    ].head(10)
)