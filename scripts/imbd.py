import pandas as pd

# Load required columns
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

ratings = pd.read_csv(
    "data/raw/title.ratings.tsv",
    sep="\t"
)

# Merge datasets
imdb = basics.merge(ratings, on="tconst")

# Keep only Movies and TV Series
imdb = imdb[
    imdb["titleType"].isin(["movie", "tvSeries"])
]

# Remove duplicates
imdb.drop_duplicates(inplace=True)

# Convert year
imdb["startYear"] = pd.to_numeric(
    imdb["startYear"],
    errors="coerce"
)

# Remove rows with missing year
imdb.dropna(subset=["startYear"], inplace=True)

# Remove extra spaces
imdb["primaryTitle"] = imdb["primaryTitle"].str.strip()

# Save
imdb.to_csv(
    "data/cleaned/imdb_cleaned_v1.csv",
    index=False
)

print("IMDb dataset cleaned successfully.")