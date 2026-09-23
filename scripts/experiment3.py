# ============================================================
# EXPERIMENT 3
# Exploratory Data Analysis & Statistical Analysis
# Netflix Data Science Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/cleaned/merged_v1.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDescriptive Statistics:")
print(df.describe())


# ============================================================
# 3. CLASS BALANCE
# ============================================================

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

print(df["type"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="type")

plt.title("Netflix Content Type Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ============================================================
# 4. IMDb RATING DISTRIBUTION
# ============================================================

ratings = df["averageRating"].dropna()

print("\n" + "=" * 60)
print("IMDb RATING DISTRIBUTION")
print("=" * 60)

print("Number of ratings:", len(ratings))
print("Mean:", ratings.mean())
print("Median:", ratings.median())
print("Standard Deviation:", ratings.std())

plt.figure(figsize=(8, 5))

sns.histplot(
    ratings,
    bins=30,
    kde=True
)

plt.title("Distribution of IMDb Ratings")
plt.xlabel("IMDb Rating")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ============================================================
# 5. IMDb RATING BOXPLOT
# ============================================================

plt.figure(figsize=(8, 4))

sns.boxplot(x=ratings)

plt.title("Boxplot of IMDb Ratings")
plt.xlabel("IMDb Rating")

plt.tight_layout()
plt.show()


# ============================================================
# 6. GAUSSIAN / NORMAL DISTRIBUTION FIT
# ============================================================

print("\n" + "=" * 60)
print("GAUSSIAN DISTRIBUTION FIT - IMDb RATINGS")
print("=" * 60)

# Fit Normal distribution
mu, sigma = stats.norm.fit(ratings)

print("Mean:", mu)
print("Standard Deviation:", sigma)


# Create fitted Gaussian curve
x = np.linspace(
    ratings.min(),
    ratings.max(),
    200
)

normal_pdf = stats.norm.pdf(
    x,
    mu,
    sigma
)

plt.figure(figsize=(8, 5))

plt.hist(
    ratings,
    bins=30,
    density=True,
    alpha=0.6,
    label="Observed IMDb Ratings"
)

plt.plot(
    x,
    normal_pdf,
    linewidth=2,
    label="Fitted Gaussian Distribution"
)

plt.title("Gaussian Distribution Fit - IMDb Ratings")
plt.xlabel("IMDb Rating")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 7. GAUSSIAN GOODNESS-OF-FIT TEST
# Kolmogorov-Smirnov Test
# ============================================================

ks_stat, gaussian_p_value = stats.kstest(
    ratings,
    "norm",
    args=(mu, sigma)
)

print("\nGaussian Goodness-of-Fit Test")

print("KS Statistic:", ks_stat)
print("P-value:", gaussian_p_value)

if gaussian_p_value > 0.05:
    print(
        "Conclusion: Gaussian distribution is "
        "a reasonable fit."
    )
else:
    print(
        "Conclusion: Gaussian distribution is "
        "not a good fit."
    )


# ============================================================
# 8. FIT OTHER DISTRIBUTIONS TO IMDb RATINGS
# ============================================================

print("\n" + "=" * 60)
print("COMPARING DISTRIBUTIONS FOR IMDb RATINGS")
print("=" * 60)


# Normal Distribution
normal_params = stats.norm.fit(ratings)


# Log-normal Distribution
lognormal_params = stats.lognorm.fit(
    ratings,
    floc=0
)


# Gamma Distribution
gamma_params = stats.gamma.fit(
    ratings,
    floc=0
)


# Beta Distribution
# Scale IMDb ratings strictly between 0 and 1
ratings_scaled = (
    (ratings - ratings.min()) /
    (ratings.max() - ratings.min())
)

epsilon = 1e-6

ratings_scaled = ratings_scaled.clip(
    epsilon,
    1 - epsilon
)

beta_params = stats.beta.fit(
    ratings_scaled,
    floc=0,
    fscale=1
)


print("\nDistribution Parameters:")

print("\nNormal:")
print(normal_params)

print("\nLog-normal:")
print(lognormal_params)

print("\nGamma:")
print(gamma_params)

print("\nBeta:")
print(beta_params)


# ============================================================
# 9. AIC FUNCTION
# ============================================================

def calculate_aic(data, distribution, params):

    log_likelihood = np.sum(
        distribution.logpdf(data, *params)
    )

    k = len(params)

    aic = 2 * k - 2 * log_likelihood

    return aic


# ============================================================
# 10. AIC COMPARISON - IMDb RATINGS
# ============================================================

aic_normal = calculate_aic(
    ratings,
    stats.norm,
    normal_params
)

aic_lognormal = calculate_aic(
    ratings,
    stats.lognorm,
    lognormal_params
)

aic_gamma = calculate_aic(
    ratings,
    stats.gamma,
    gamma_params
)

aic_beta = calculate_aic(
    ratings_scaled,
    stats.beta,
    beta_params
)


print("\n" + "=" * 60)
print("AIC COMPARISON - IMDb RATINGS")
print("=" * 60)

print("Normal AIC:", aic_normal)
print("Log-normal AIC:", aic_lognormal)
print("Gamma AIC:", aic_gamma)
print("Beta AIC:", aic_beta)


aic_values = {
    "Normal": aic_normal,
    "Log-normal": aic_lognormal,
    "Gamma": aic_gamma,
    "Beta": aic_beta
}

best_distribution = min(
    aic_values,
    key=aic_values.get
)

print("\nBest fitting distribution based on AIC:")
print(best_distribution)


# ============================================================
# 11. RUNTIME DISTRIBUTION
# ============================================================

runtime = df["runtimeMinutes"].dropna()

print("\n" + "=" * 60)
print("RUNTIME DISTRIBUTION")
print("=" * 60)

print("Number of runtime values:", len(runtime))
print("Mean Runtime:", runtime.mean())
print("Median Runtime:", runtime.median())
print("Standard Deviation:", runtime.std())


plt.figure(figsize=(8, 5))

sns.histplot(
    runtime,
    bins=40,
    kde=True
)

plt.title("Distribution of Runtime")
plt.xlabel("Runtime (Minutes)")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.show()


# ============================================================
# 12. RUNTIME BOXPLOT
# ============================================================

plt.figure(figsize=(8, 4))

sns.boxplot(x=runtime)

plt.title("Boxplot of Runtime")
plt.xlabel("Runtime (Minutes)")

plt.tight_layout()
plt.show()


# ============================================================
# 13. FIT DISTRIBUTIONS TO RUNTIME
# ============================================================

runtime_normal_params = stats.norm.fit(runtime)

runtime_lognormal_params = stats.lognorm.fit(
    runtime,
    floc=0
)

runtime_gamma_params = stats.gamma.fit(
    runtime,
    floc=0
)


# ============================================================
# 14. RUNTIME AIC COMPARISON
# ============================================================

runtime_aic_normal = calculate_aic(
    runtime,
    stats.norm,
    runtime_normal_params
)

runtime_aic_lognormal = calculate_aic(
    runtime,
    stats.lognorm,
    runtime_lognormal_params
)

runtime_aic_gamma = calculate_aic(
    runtime,
    stats.gamma,
    runtime_gamma_params
)


print("\n" + "=" * 60)
print("AIC COMPARISON - RUNTIME")
print("=" * 60)

print("Normal:", runtime_aic_normal)
print("Log-normal:", runtime_aic_lognormal)
print("Gamma:", runtime_aic_gamma)


runtime_aic_values = {
    "Normal": runtime_aic_normal,
    "Log-normal": runtime_aic_lognormal,
    "Gamma": runtime_aic_gamma
}

best_runtime_distribution = min(
    runtime_aic_values,
    key=runtime_aic_values.get
)

print(
    "\nBest fitting distribution for Runtime:",
    best_runtime_distribution
)


# ============================================================
# 15. NUMBER OF VOTES - POISSON CHECK
# ============================================================

votes = df["numVotes"].dropna()

print("\n" + "=" * 60)
print("POISSON DISTRIBUTION CHECK - NUMBER OF VOTES")
print("=" * 60)

votes_mean = votes.mean()
votes_variance = votes.var()

print("Mean:", votes_mean)
print("Variance:", votes_variance)

variance_mean_ratio = votes_variance / votes_mean

print(
    "Variance / Mean Ratio:",
    variance_mean_ratio
)

if abs(votes_variance - votes_mean) / votes_mean < 0.20:

    print(
        "Mean and variance are relatively close. "
        "Poisson distribution may be considered."
    )

else:

    print(
        "Variance is considerably different from mean. "
        "Poisson distribution is not a good fit."
    )


# ============================================================
# 16. CORRELATION HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_columns = [
    "release_year",
    "startYear",
    "runtimeMinutes",
    "averageRating",
    "numVotes"
]

correlation_data = df[numeric_columns]

correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 17. MOVIE VS TV SHOW IMDb RATINGS
# ============================================================

movies = df[
    df["type"] == "Movie"
]["averageRating"].dropna()

tv_shows = df[
    df["type"] == "TV Show"
]["averageRating"].dropna()


print("\n" + "=" * 60)
print("MOVIE VS TV SHOW IMDb RATINGS")
print("=" * 60)

print("Movie Mean:", movies.mean())
print("TV Show Mean:", tv_shows.mean())

print("Movie Count:", len(movies))
print("TV Show Count:", len(tv_shows))


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="type",
    y="averageRating"
)

plt.title(
    "IMDb Rating Comparison: Movies vs TV Shows"
)

plt.xlabel("Content Type")
plt.ylabel("IMDb Rating")

plt.tight_layout()
plt.show()


# ============================================================
# 18. HYPOTHESIS TESTING
# Independent Two-Sample T-Test
# ============================================================

print("\n" + "=" * 60)
print("HYPOTHESIS TESTING")
print("=" * 60)

print("\nNull Hypothesis (H0):")
print(
    "There is no significant difference between "
    "the mean IMDb ratings of Movies and TV Shows."
)

print("\nAlternative Hypothesis (H1):")
print(
    "There is a significant difference between "
    "the mean IMDb ratings of Movies and TV Shows."
)

alpha = 0.05


# Welch's independent t-test
t_stat, ttest_p_value = stats.ttest_ind(
    movies,
    tv_shows,
    equal_var=False
)


print("\nSignificance Level (alpha):", alpha)

print("T-statistic:", t_stat)

print("P-value:", ttest_p_value)


# ============================================================
# 19. HYPOTHESIS TEST CONCLUSION
# ============================================================

print("\nHypothesis Test Interpretation:")

if ttest_p_value < alpha:

    print("P-value is less than 0.05.")
    print("Reject the null hypothesis (H0).")

    print(
        "There is a statistically significant "
        "difference between the mean IMDb ratings "
        "of Movies and TV Shows."
    )

else:

    print(
        "P-value is greater than or equal to 0.05."
    )

    print(
        "Fail to reject the null hypothesis (H0)."
    )

    print(
        "There is insufficient statistical evidence "
        "to conclude that the mean IMDb ratings "
        "of Movies and TV Shows are different."
    )


# ============================================================
# 20. FINAL EDA INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("EDA INSIGHTS")
print("=" * 60)

print("""
1. The dataset contains both Movies and TV Shows.

2. IMDb ratings are concentrated around the central
   rating range.

3. The IMDb rating distribution was tested against
   a Gaussian distribution using the Kolmogorov-Smirnov test.

4. Multiple probability distributions were compared
   using AIC to identify the best fitting distribution
   for IMDb ratings.

5. Runtime shows a right-skewed distribution with
   potential outliers at higher runtime values.

6. Normal, Log-normal and Gamma distributions were
   compared for runtime using AIC.

7. Number of votes was checked for suitability of a
   Poisson distribution by comparing its mean and variance.

8. The correlation heatmap shows relationships between
   numerical features.

9. Movie and TV Show IMDb ratings were compared using
   an independent two-sample t-test.

10. The hypothesis-test conclusion is based on the
    obtained p-value and significance level of 0.05.
""")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3 COMPLETED")
print("=" * 60)
# ============================================================
# POISSON DISTRIBUTION FIT - NUMBER OF VOTES
# ============================================================

print("\n" + "=" * 60)
print("POISSON DISTRIBUTION FIT - NUMBER OF VOTES")
print("=" * 60)

votes = df["numVotes"].dropna()

# Poisson parameter lambda is the mean
lambda_poisson = votes.mean()

print("Poisson Lambda (mean):", lambda_poisson)

# Check mean and variance
print("Mean:", votes.mean())
print("Variance:", votes.var())
print("Variance / Mean:", votes.var() / votes.mean())


# ============================================================
# POISSON GRAPH
# ============================================================

# Since numVotes contains extremely large values,
# displaying every possible count is not practical.
# We use the range up to the 99th percentile.

upper_limit = int(votes.quantile(0.99))

votes_plot = votes[
    votes <= upper_limit
]

x_poisson = np.arange(
    0,
    upper_limit + 1
)

# Poisson probability
poisson_pmf = stats.poisson.pmf(
    x_poisson,
    lambda_poisson
)


plt.figure(figsize=(10, 6))

plt.hist(
    votes_plot,
    bins=50,
    density=True,
    alpha=0.6,
    label="Observed Number of Votes"
)

plt.plot(
    x_poisson,
    poisson_pmf,
    linewidth=2,
    label="Fitted Poisson Distribution"
)

plt.title(
    "Poisson Distribution Fit - Number of Votes"
)

plt.xlabel("Number of Votes")
plt.ylabel("Probability / Density")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# POISSON CONCLUSION
# ============================================================

if abs(votes.var() - votes.mean()) / votes.mean() < 0.20:

    print(
        "Conclusion: Mean and variance are relatively close."
    )

    print(
        "Poisson distribution may be suitable."
    )

else:

    print(
        "Conclusion: Variance is much larger than the mean."
    )

    print(
        "Poisson distribution is not a suitable fit."
    )
    
    # ============================================================
# 16B. NEGATIVE BINOMIAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("NEGATIVE BINOMIAL DISTRIBUTION - NUMBER OF VOTES")
print("=" * 60)

votes = df["numVotes"].dropna()

# Remove zero values for fitting a continuous approximation
# because the vote distribution is extremely skewed
votes_positive = votes[votes > 0]

mean_votes = votes_positive.mean()
variance_votes = votes_positive.var()

print("Mean:", mean_votes)
print("Variance:", variance_votes)


# Estimate Negative Binomial parameters
# Variance = mean + mean^2 / r

r = (mean_votes ** 2) / (variance_votes - mean_votes)

p = r / (r + mean_votes)

print("\nNegative Binomial Parameters:")
print("r:", r)
print("p:", p)


# ============================================================
# PLOT
# ============================================================

upper_limit = int(votes_positive.quantile(0.99))

votes_plot = votes_positive[
    votes_positive <= upper_limit
]

plt.figure(figsize=(10, 6))

plt.hist(
    votes_plot,
    bins=50,
    density=True,
    alpha=0.6,
    label="Observed Number of Votes"
)

x = np.arange(
    1,
    upper_limit + 1
)

nb_pmf = stats.nbinom.pmf(
    x,
    r,
    p
)

plt.plot(
    x,
    nb_pmf,
    linewidth=2,
    label="Fitted Negative Binomial"
)

plt.title(
    "Negative Binomial Distribution Fit - Number of Votes"
)

plt.xlabel("Number of Votes")
plt.ylabel("Probability / Density")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# CONCLUSION
# ============================================================

print("\nConclusion:")

print(
    "The Number of Votes data is highly overdispersed, "
    "as its variance is much greater than its mean."
)

print(
    "Therefore, the Poisson distribution is not suitable."
)

print(
    "A Negative Binomial distribution is considered "
    "more appropriate for modelling the overdispersed "
    "count data."
)
# ============================================================
# 16B. NEGATIVE BINOMIAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("NEGATIVE BINOMIAL DISTRIBUTION - NUMBER OF VOTES")
print("=" * 60)

votes = df["numVotes"].dropna()

# Use the complete dataset, including zero votes
mean_votes = votes.mean()
variance_votes = votes.var()

print("Mean:", mean_votes)
print("Variance:", variance_votes)

print(
    "Variance / Mean Ratio:",
    variance_votes / mean_votes
)


# ============================================================
# ESTIMATE NEGATIVE BINOMIAL PARAMETERS
# ============================================================

# For Negative Binomial:
#
# Variance = mean + mean^2 / r
#
# Therefore:
#
# r = mean^2 / (variance - mean)

r = (
    mean_votes ** 2
    / (variance_votes - mean_votes)
)

p = r / (r + mean_votes)

print("\nNegative Binomial Parameters:")
print("r:", r)
print("p:", p)


# ============================================================
# POISSON PARAMETERS
# ============================================================

lambda_poisson = mean_votes

print("\nPoisson Lambda:")
print(lambda_poisson)


# ============================================================
# AIC COMPARISON
# ============================================================

# Poisson log likelihood
poisson_log_likelihood = np.sum(
    stats.poisson.logpmf(
        votes,
        lambda_poisson
    )
)

poisson_aic = (
    2 * 1
    - 2 * poisson_log_likelihood
)


# Negative Binomial log likelihood
nb_log_likelihood = np.sum(
    stats.nbinom.logpmf(
        votes,
        r,
        p
    )
)

nb_aic = (
    2 * 2
    - 2 * nb_log_likelihood
)


print("\n" + "=" * 60)
print("POISSON VS NEGATIVE BINOMIAL")
print("=" * 60)

print("Poisson AIC:", poisson_aic)
print("Negative Binomial AIC:", nb_aic)


if nb_aic < poisson_aic:

    print(
        "\nConclusion: Negative Binomial has a lower AIC "
        "than Poisson."
    )

    print(
        "Therefore, Negative Binomial provides a better "
        "fit for the Number of Votes data."
    )

else:

    print(
        "\nConclusion: Poisson has a lower AIC."
    )

    print(
        "However, the extreme overdispersion should "
        "still be considered."
    )


# ============================================================
# NEGATIVE BINOMIAL VISUALIZATION
# ============================================================

# The full range goes up to millions of votes, so for
# visualization we display the central 99% of observations.

upper_limit = int(
    votes.quantile(0.99)
)

votes_plot = votes[
    votes <= upper_limit
]


# Create histogram bins
bins = 50

plt.figure(figsize=(10, 6))

plt.hist(
    votes_plot,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Observed Number of Votes"
)


# For a discrete distribution, plot probability
# at representative values.

x_nb = np.linspace(
    0,
    upper_limit,
    500
)

nb_probability = stats.nbinom.pmf(
    np.round(x_nb).astype(int),
    r,
    p
)

plt.plot(
    x_nb,
    nb_probability,
    linewidth=2,
    label="Fitted Negative Binomial"
)

plt.title(
    "Negative Binomial Distribution Fit - Number of Votes"
)

plt.xlabel("Number of Votes")
plt.ylabel("Probability / Density")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL INTERPRETATION - NUMBER OF VOTES")
print("=" * 60)

print(
    "The Number of Votes variable is count data."
)

print(
    "Its variance is much greater than its mean, "
    "indicating strong overdispersion."
)

print(
    "Therefore, the basic Poisson distribution is "
    "not appropriate."
)

print(
    "The Negative Binomial distribution is evaluated "
    "as an alternative model for the overdispersed "
    "count data."
)
# ============================================================
# CHI-SQUARE TEST OF INDEPENDENCE
# ============================================================

print("\n" + "=" * 60)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 60)


# We test whether Netflix content type
# is associated with content rating.

# Create contingency table
contingency_table = pd.crosstab(
    df["type"],
    df["rating"]
)


print("\nContingency Table:")
print(contingency_table)


# Perform Chi-square test
chi2_stat, p_value, degrees_freedom, expected = chi2_contingency(
    contingency_table
)


print("\nChi-square statistic:")
print(round(chi2_stat, 6))


print("\nDegrees of freedom:")
print(degrees_freedom)


print("\nP-value:")
print(p_value)


# Significance level
alpha = 0.05


print("\nSignificance level:")
print(alpha)


# Hypothesis decision
if p_value < alpha:

    print("\nDecision: Reject H0")

    print(
        "There is a statistically significant association "
        "between content type and content rating."
    )

else:

    print("\nDecision: Fail to reject H0")

    print(
        "There is no statistically significant association "
        "between content type and content rating."
    )


print("\nHypotheses:")

print(
    "H0: Content type and content rating are independent."
)

print(
    "H1: Content type and content rating are associated."
)