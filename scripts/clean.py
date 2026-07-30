import pandas as pd

# Load dataset
netflix = pd.read_csv("data/raw/netflix_titles.csv")

# Remove duplicate rows
netflix.drop_duplicates(inplace=True)

# Remove leading/trailing spaces
netflix["title"] = netflix["title"].str.strip()

# Replace missing values
netflix["director"] = netflix["director"].fillna("Unknown")
netflix["cast"] = netflix["cast"].fillna("Unknown")
netflix["country"] = netflix["country"].fillna("Unknown")

# Remove rows where title is missing
netflix.dropna(subset=["title"], inplace=True)

# Convert release year to integer
netflix["release_year"] = netflix["release_year"].astype(int)

# Save cleaned dataset
netflix.to_csv(
    "data/cleaned/netflix_cleaned_v1.csv",
    index=False
)

print("Netflix dataset cleaned successfully.")