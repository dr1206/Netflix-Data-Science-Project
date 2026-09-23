# ============================================================
# EXPERIMENT 3
# EXPLORATORY DATA ANALYSIS & STATISTICAL ANALYSIS
# Netflix + IMDb Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    ttest_ind,
    normaltest,
    skew,
    kurtosis
)

from sklearn.preprocessing import StandardScaler


print("=" * 70)
print(" EXPLORATORY DATA ANALYSIS & STATISTICAL ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/cleaned/merged_v1.csv",
    keep_default_na=False
)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA PROFILING")
print("=" * 70)


print("\nData Types:")
print(df.dtypes)


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Rows:")
print(df.duplicated().sum())


print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numeric_columns = [
    "release_year",
    "startYear",
    "runtimeMinutes",
    "averageRating",
    "numVotes"
]


numeric_df = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# OBJECTIVE 1
# DISTRIBUTION OF CLASSES AND FEATURES
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 1: DISTRIBUTION OF CLASSES AND FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# 2.1 Movies vs TV Shows
# ------------------------------------------------------------

type_counts = df["type"].value_counts()

print("\nMovies vs TV Shows:")
print(type_counts)


plt.figure(figsize=(7, 5))

type_counts.plot(kind="bar")

plt.title("Distribution of Movies and TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 2.2 DISTRIBUTION ANALYSIS OF ALL NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION ANALYSIS OF NUMERICAL FEATURES")
print("=" * 70)


for column in numeric_columns:

    data = numeric_df[column].dropna()

    print("\n" + "-" * 60)
    print("Feature:", column)
    print("-" * 60)

    print("Count:", len(data))
    print("Mean:", data.mean())
    print("Median:", data.median())
    print("Mode:", data.mode()[0])
    print("Minimum:", data.min())
    print("Maximum:", data.max())
    print("Range:", data.max() - data.min())
    print("Variance:", data.var())
    print("Standard Deviation:", data.std())

    skew_value = data.skew()
    kurtosis_value = data.kurtosis()

    print("Skewness:", skew_value)
    print("Kurtosis:", kurtosis_value)


    # --------------------------------------------------------
    # Skewness Interpretation
    # --------------------------------------------------------

    if abs(skew_value) < 0.5:

        print("Skewness Interpretation: Approximately symmetric")

    elif skew_value >= 0.5 and skew_value < 1:

        print("Skewness Interpretation: Moderately right-skewed")

    elif skew_value >= 1:

        print("Skewness Interpretation: Highly right-skewed")

    elif skew_value <= -0.5 and skew_value > -1:

        print("Skewness Interpretation: Moderately left-skewed")

    elif skew_value <= -1:

        print("Skewness Interpretation: Highly left-skewed")


    # --------------------------------------------------------
    # Normality Test
    # --------------------------------------------------------

    statistic, p_value = normaltest(data)

    print("Normality Test Statistic:", statistic)
    print("Normality Test P-value:", p_value)

    alpha_normality = 0.05

    if p_value < alpha_normality:

        print(
            "Normality Result: Not normally distributed "
            "(Reject H0)"
        )

    else:

        print(
            "Normality Result: No significant evidence "
            "against normality (Fail to Reject H0)"
        )


    # --------------------------------------------------------
    # Histogram + KDE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data,
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ============================================================
# 2.3 Q-Q PLOTS FOR NORMALITY
# ============================================================

print("\n" + "=" * 70)
print("Q-Q PLOTS FOR NORMALITY")
print("=" * 70)


import scipy.stats as stats


for column in numeric_columns:

    data = numeric_df[column].dropna()

    plt.figure(figsize=(6, 6))

    stats.probplot(
        data,
        dist="norm",
        plot=plt
    )

    plt.title(f"Q-Q Plot: {column}")

    plt.tight_layout()
    plt.show()


# ============================================================
# 2.4 RELEASE YEAR DISTRIBUTION
# ============================================================

year_counts = (
    df["release_year"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(12, 5))

plt.plot(
    year_counts.index,
    year_counts.values
)

plt.title("Netflix Content by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 2
# CENTRAL TENDENCY AND SPREAD
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 2: CENTRAL TENDENCY AND SPREAD")
print("=" * 70)


# ------------------------------------------------------------
# 3.1 IMDb Rating Statistics
# ------------------------------------------------------------

print("\nIMDb Rating Statistics:")

print("Mean   :", numeric_df["averageRating"].mean())
print("Median :", numeric_df["averageRating"].median())
print("Mode   :", numeric_df["averageRating"].mode()[0])


# ------------------------------------------------------------
# 3.2 Number of Votes Statistics
# ------------------------------------------------------------

print("\nNumber of Votes Statistics:")

print("Mean   :", numeric_df["numVotes"].mean())
print("Median :", numeric_df["numVotes"].median())
print("Mode   :", numeric_df["numVotes"].mode()[0])


# ------------------------------------------------------------
# 3.3 IMDb Rating Spread
# ------------------------------------------------------------

print("\nIMDb Rating Spread:")

print(
    "Minimum :",
    numeric_df["averageRating"].min()
)

print(
    "Maximum :",
    numeric_df["averageRating"].max()
)

print(
    "Range   :",
    numeric_df["averageRating"].max()
    - numeric_df["averageRating"].min()
)

print(
    "Std Dev :",
    numeric_df["averageRating"].std()
)


# ------------------------------------------------------------
# 3.4 IMDb Rating Box Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    y=numeric_df["averageRating"]
)

plt.title("Box Plot of IMDb Ratings")
plt.ylabel("IMDb Rating")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.5 Movies vs TV Shows Rating Box Plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="type",
    y="averageRating"
)

plt.title("IMDb Rating Distribution: Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("IMDb Rating")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.6 Runtime Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    numeric_df["runtimeMinutes"].dropna(),
    bins=20,
    kde=True
)

plt.title("Distribution of Runtime")
plt.xlabel("Runtime (Minutes)")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 3
# FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 3: FEATURE COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# 4.1 Standardize Numerical Features
# ------------------------------------------------------------

scaler = StandardScaler()


scaled_data = scaler.fit_transform(
    numeric_df.dropna()
)


scaled_df = pd.DataFrame(
    scaled_data,
    columns=numeric_columns
)


print("\nStandardized Feature Statistics:")
print(scaled_df.describe())


# ------------------------------------------------------------
# 4.2 Standardized Box Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=scaled_df
)

plt.title("Standardized Comparison of Numerical Features")
plt.xlabel("Features")
plt.ylabel("Standardized Value (Z-score)")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 4
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 4: CORRELATION ANALYSIS")
print("=" * 70)


correlation_matrix = numeric_df.corr()


print("\nCorrelation Matrix:")
print(correlation_matrix)


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 5
# POISSON DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 5: POISSON DISTRIBUTION CHECK")
print("=" * 70)


votes = numeric_df["numVotes"].dropna()


mean_votes = votes.mean()
variance_votes = votes.var()


print("\nFeature: numVotes")

print("Mean:", mean_votes)

print("Variance:", variance_votes)

print(
    "Variance / Mean:",
    variance_votes / mean_votes
)


# ------------------------------------------------------------
# Poisson Interpretation
# ------------------------------------------------------------

if abs(variance_votes - mean_votes) / mean_votes < 0.20:

    print(
        "\nPoisson Interpretation:"
        "\nMean and variance are relatively close."
        "\nThe data may be compatible with a Poisson-like distribution."
    )

else:

    print(
        "\nPoisson Interpretation:"
        "\nMean and variance are substantially different."
        "\nThe data does not appear Poisson-like."
    )


# ------------------------------------------------------------
# Log-transformed Vote Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    np.log1p(votes),
    bins=30,
    kde=True
)

plt.title("Log-Transformed Distribution of IMDb Votes")
plt.xlabel("log(1 + Number of Votes)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 6
# STATISTICAL HYPOTHESIS TESTING
# ============================================================

print("\n" + "=" * 70)
print("OBJECTIVE 6: STATISTICAL HYPOTHESIS TESTING")
print("=" * 70)


# ------------------------------------------------------------
# Compare IMDb Ratings of Movies and TV Shows
# ------------------------------------------------------------

movies = df[
    df["type"] == "Movie"
]["averageRating"]


tv_shows = df[
    df["type"] == "TV Show"
]["averageRating"]


# Convert to numeric and remove invalid values

movies = pd.to_numeric(
    movies,
    errors="coerce"
).dropna()


tv_shows = pd.to_numeric(
    tv_shows,
    errors="coerce"
).dropna()


print("\nNumber of Movie Ratings:", len(movies))

print(
    "Number of TV Show Ratings:",
    len(tv_shows)
)


print(
    "\nMean Movie Rating:",
    movies.mean()
)


print(
    "Mean TV Show Rating:",
    tv_shows.mean()
)


# ============================================================
# INDEPENDENT TWO-SAMPLE WELCH'S T-TEST
# ============================================================

t_statistic, p_value = ttest_ind(
    movies,
    tv_shows,
    equal_var=False
)


print("\nT-Test Results")
print("-" * 40)

print(
    "T-statistic:",
    t_statistic
)

print(
    "P-value:",
    p_value
)


# ============================================================
# HYPOTHESIS DECISION
# ============================================================

alpha = 0.05


print(
    "\nSignificance Level (alpha):",
    alpha
)


if p_value < alpha:

    print(
        "\nResult: Reject the Null Hypothesis."
    )

    print(
        "There is a statistically significant difference "
        "between the IMDb ratings of Movies and TV Shows."
    )

else:

    print(
        "\nResult: Fail to Reject the Null Hypothesis."
    )

    print(
        "There is no statistically significant difference "
        "between the IMDb ratings of Movies and TV Shows."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(" EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)