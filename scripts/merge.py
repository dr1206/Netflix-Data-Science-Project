# ==========================================
# 04_merge_data.py
# Purpose: Merge Netflix and IMDb datasets
# ==========================================

import pandas as pd

print("=" * 60)
print("LOADING CLEANED DATASETS")
print("=" * 60)

# Load cleaned datasets
netflix = pd.read_csv("data/cleaned/netflix_cleaned_v1.csv")
imdb = pd.read_csv("data/cleaned/imdb_cleaned_v1.csv")

print("Netflix Shape :", netflix.shape)
print("IMDb Shape    :", imdb.shape)

# ---------------------------------------------------
# Standardize titles before merging
# ---------------------------------------------------

netflix["title"] = (
    netflix["title"]
    .astype(str)
    .str.strip()
    .str.lower()
)

imdb["primaryTitle"] = (
    imdb["primaryTitle"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------
# Merge datasets
# ---------------------------------------------------

print("\nMerging datasets...")

merged = pd.merge(
    netflix,
    imdb,
    left_on=["title", "release_year"],
    right_on=["primaryTitle", "startYear"],
    how="left"
)

print("Merge Completed Successfully!")

# ---------------------------------------------------
# Fill Missing IMDb Values
# ---------------------------------------------------

merged["averageRating"] = merged["averageRating"].fillna(0)

merged["numVotes"] = merged["numVotes"].fillna(0)

merged["numVotes"] = merged["numVotes"].astype(int)

# ---------------------------------------------------
# Display Merge Summary
# ---------------------------------------------------

print("\nMerged Dataset Shape")
print(merged.shape)

print("\nMissing IMDb Ratings")
print(merged[["averageRating", "numVotes"]].isnull().sum())

print("\nFirst 10 Rows")
print(
    merged[
        [
            "title",
            "release_year",
            "primaryTitle",
            "startYear",
            "averageRating",
            "numVotes",
        ]
    ].head(10)
)

# ---------------------------------------------------
# Save Dataset
# ---------------------------------------------------

merged.to_csv(
    "data/cleaned/merged_v1.csv",
    index=False
)

print("\nMerged dataset saved successfully!")
print("Location : data/cleaned/merged_v1.csv")