# ==========================================
# 04_merge_data.py
# Purpose: Merge Netflix and IMDb datasets
# ==========================================

import pandas as pd

print("=" * 60)
print("LOADING CLEANED DATASETS")
print("=" * 60)

# ---------------------------------------------------
# Load cleaned datasets
# ---------------------------------------------------

netflix = pd.read_csv("data/cleaned/netflix_cleaned_v1.csv")
imdb = pd.read_csv("data/cleaned/imdb_cleaned_v1.csv")

print("Netflix Shape :", netflix.shape)
print("IMDb Shape    :", imdb.shape)

# ---------------------------------------------------
# Standardize Titles
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
# Convert IMDb Numeric Columns
# ---------------------------------------------------

imdb["startYear"] = pd.to_numeric(imdb["startYear"], errors="coerce")
imdb["runtimeMinutes"] = pd.to_numeric(imdb["runtimeMinutes"], errors="coerce")
imdb["averageRating"] = pd.to_numeric(imdb["averageRating"], errors="coerce")
imdb["numVotes"] = pd.to_numeric(imdb["numVotes"], errors="coerce")

# ---------------------------------------------------
# Merge
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

print("\nHandling Missing Values...")

# =====================================================
# Netflix Columns
# =====================================================

merged["director"] = merged["director"].fillna(
    merged["director"].mode()[0]
)

merged["cast"] = merged["cast"].fillna(
    merged["cast"].mode()[0]
)

merged["country"] = merged["country"].fillna(
    merged["country"].mode()[0]
)

merged["date_added"] = merged["date_added"].fillna(
    merged["date_added"].mode()[0]
)

merged["rating"] = merged["rating"].fillna(
    merged["rating"].mode()[0]
)

merged["duration"] = merged["duration"].fillna(
    merged["duration"].mode()[0]
)

# =====================================================
# IMDb Columns
# =====================================================

merged["primaryTitle"] = merged["primaryTitle"].fillna(
    merged["title"]
)

merged["startYear"] = merged["startYear"].fillna(
    merged["release_year"]
)

merged["titleType"] = merged["titleType"].fillna(
    merged["type"].replace({
        "Movie": "movie",
        "TV Show": "tvSeries"
    })
)

merged["genres"] = merged["genres"].fillna(
    merged["listed_in"]
)



# ---------------------------------------------------
# Convert merged numeric columns
# ---------------------------------------------------

merged["runtimeMinutes"] = pd.to_numeric(
    merged["runtimeMinutes"],
    errors="coerce"
)

merged["averageRating"] = pd.to_numeric(
    merged["averageRating"],
    errors="coerce"
)

merged["numVotes"] = pd.to_numeric(
    merged["numVotes"],
    errors="coerce"
)

# ---------------------------------------------------
# Fill numeric values
# ---------------------------------------------------

runtime_median = merged["runtimeMinutes"].median()
rating_median = merged["averageRating"].median()

merged["runtimeMinutes"] = merged["runtimeMinutes"].fillna(
    runtime_median
).astype(int)

merged["averageRating"] = merged["averageRating"].fillna(
    rating_median
)

merged["numVotes"] = merged["numVotes"].fillna(
    0
).astype(int)

merged["startYear"] = merged["startYear"].astype(int)

# Fill missing IMDb IDs
merged["tconst"] = merged["tconst"].fillna("IMDb_Not_Found")
merged["tconst"] = merged["tconst"].replace("NA", "IMDb_Not_Found")
merged["tconst"] = merged["tconst"].replace("", "IMDb_Not_Found")




# =====================================================
# Verify
# =====================================================

print("\nRemaining Missing Values")
print("-" * 40)
print(merged.isnull().sum())

print("\nMerged Dataset Shape")
print(merged.shape)

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
# Save
# ---------------------------------------------------

merged.to_csv(
    "data/cleaned/merged_v1.csv",
    index=False
)

print("\nMerged dataset saved successfully!")
print("Location : data/cleaned/merged_v1.csv")