# ============================================================
# EXPERIMENT 5
# EXPLAINABLE AI AND FAIRNESS
#
# SHAP + LIME + FAIRLEARN
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    MetricFrame
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. EXPERIMENT INFORMATION
# ============================================================

print("=" * 60)
print("EXPERIMENT 5")
print("EXPLAINABLE AI AND FAIRNESS")
print("=" * 60)


# ============================================================
# 2. CREATE RESULTS DIRECTORY
# ============================================================

results_dir = "results/experiment5"

os.makedirs(
    results_dir,
    exist_ok=True
)


print("\nResults will be saved in:")
print(results_dir)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("DATASET LOADING")
print("=" * 60)


df = pd.read_csv(
    "data/cleaned/merged_v1.csv"
)


print("Dataset shape:", df.shape)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "release_year",
    "runtimeMinutes",
    "averageRating",
    "numVotes"
]

target = "type"


X = df[features].copy()

y = df[target].copy()


print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

print("\nTarget classes:")
print(y.unique())


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. LOAD BEST MODEL FROM EXPERIMENT 4
# ============================================================

print("\n" + "=" * 60)
print("MODEL LOADING")
print("=" * 60)


model_path = "models/best_model.pkl"


if not os.path.exists(model_path):

    raise FileNotFoundError(
        "best_model.pkl was not found. "
        "Run Experiment 4 first."
    )


model = joblib.load(
    model_path
)


print("Model loaded successfully:")
print(type(model))


# ============================================================
# 7. MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PREDICTIONS")
print("=" * 60)


y_pred = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Test Accuracy:",
    round(test_accuracy, 6)
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 8. SHAP EXPLAINABILITY
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)


print("\nCreating SHAP TreeExplainer...")


# Random Forest is a tree-based model,
# therefore TreeExplainer is appropriate.

explainer = shap.TreeExplainer(
    model
)


# Calculate SHAP values
shap_values = explainer.shap_values(
    X_test
)


print("SHAP values calculated successfully.")


# ============================================================
# 9. HANDLE SHAP OUTPUT FORMAT
# ============================================================
#
# Different SHAP versions may return:
#
# - list of arrays
# - 2D numpy array
# - 3D numpy array
#
# We select the SHAP values corresponding
# to the TV Show class when possible.
# ============================================================

if isinstance(shap_values, list):

    if len(shap_values) == 2:

        shap_for_plot = shap_values[1]

    else:

        shap_for_plot = shap_values[0]

else:

    shap_array = np.asarray(
        shap_values
    )

    if shap_array.ndim == 3:

        # Select second class
        shap_for_plot = shap_array[:, :, 1]

    else:

        shap_for_plot = shap_array


print(
    "SHAP matrix shape:",
    np.asarray(shap_for_plot).shape
)


# ============================================================
# 10. SHAP GLOBAL SUMMARY PLOT
# ============================================================

print("\nGenerating SHAP summary plot...")


plt.figure(
    figsize=(10, 6)
)


shap.summary_plot(
    shap_for_plot,
    X_test,
    show=False
)


plt.title(
    "SHAP Global Feature Importance"
)


plt.tight_layout()


shap_summary_path = os.path.join(
    results_dir,
    "shap_summary.png"
)


plt.savefig(
    shap_summary_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP summary plot saved to:",
    shap_summary_path
)


# ============================================================
# 11. SHAP FEATURE IMPORTANCE BAR PLOT
# ============================================================

print("\nGenerating SHAP feature importance bar plot...")


plt.figure(
    figsize=(10, 6)
)


shap.summary_plot(
    shap_for_plot,
    X_test,
    plot_type="bar",
    show=False
)


plt.title(
    "SHAP Feature Importance"
)


plt.tight_layout()


shap_bar_path = os.path.join(
    results_dir,
    "shap_feature_importance.png"
)


plt.savefig(
    shap_bar_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP feature importance plot saved to:",
    shap_bar_path
)


# ============================================================
# 12. SHAP DEPENDENCE PLOT
# ============================================================

print("\nGenerating SHAP dependence plot...")


# Choose the most important feature based
# on mean absolute SHAP value.

mean_abs_shap = np.abs(
    np.asarray(shap_for_plot)
).mean(
    axis=0
)


importance_df = pd.DataFrame({

    "Feature": features,

    "Mean_Absolute_SHAP": mean_abs_shap

})


importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)


most_important_feature = (
    importance_df.iloc[0]["Feature"]
)


print(
    "Most important feature:",
    most_important_feature
)


plt.figure(
    figsize=(10, 6)
)


shap.dependence_plot(

    most_important_feature,

    shap_for_plot,

    X_test,

    show=False
)


plt.title(
    "SHAP Dependence Plot - "
    + most_important_feature
)


plt.tight_layout()


shap_dependence_path = os.path.join(
    results_dir,
    "shap_dependence.png"
)


plt.savefig(
    shap_dependence_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP dependence plot saved to:",
    shap_dependence_path
)


# ============================================================
# 13. PRINT SHAP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE")
print("=" * 60)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 14. LIME EXPLAINABILITY
# ============================================================

print("\n" + "=" * 60)
print("LIME LOCAL EXPLANATION")
print("=" * 60)


print(
    "\nCreating LIME explainer..."
)


lime_explainer = LimeTabularExplainer(

    training_data=X_train.values,

    feature_names=features,

    class_names=[
        "Movie",
        "TV Show"
    ],

    mode="classification",

    discretize_continuous=True,

    random_state=42
)


# ============================================================
# 15. SELECT A TEST INSTANCE
# ============================================================

sample_index = 0

sample = X_test.iloc[
    sample_index
].values


actual_class = y_test.iloc[
    sample_index
]


predicted_class = y_pred[
    sample_index
]


print("\nSelected test sample:")
print(X_test.iloc[sample_index])


print(
    "\nActual class:",
    actual_class
)


print(
    "Predicted class:",
    predicted_class
)


# ============================================================
# 16. LIME PREDICTION FUNCTION
# ============================================================

def lime_predict(data):

    data_df = pd.DataFrame(
        data,
        columns=features
    )

    return model.predict_proba(
        data_df
    )


# ============================================================
# 17. GENERATE LIME EXPLANATION
# ============================================================

print("\nGenerating LIME explanation...")


lime_explanation = lime_explainer.explain_instance(

    sample,

    lime_predict,

    num_features=len(features)
)


# ============================================================
# 18. PRINT LIME EXPLANATION
# ============================================================

print("\nLIME feature contributions:")

for feature, weight in lime_explanation.as_list():

    print(
        f"{feature}: {weight:.6f}"
    )


# ============================================================
# 19. SAVE LIME EXPLANATION AS HTML
# ============================================================

lime_html_path = os.path.join(
    results_dir,
    "lime_explanation.html"
)


lime_explanation.save_to_file(
    lime_html_path
)


print(
    "\nLIME explanation saved to:",
    lime_html_path
)


# ============================================================
# 20. CREATE LIME BAR PLOT
# ============================================================

lime_values = lime_explanation.as_list()


lime_features = [
    item[0]
    for item in lime_values
]


lime_weights = [
    item[1]
    for item in lime_values
]


plt.figure(
    figsize=(10, 6)
)


plt.barh(
    lime_features,
    lime_weights
)


plt.xlabel(
    "Contribution Weight"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "LIME Local Explanation"
)


plt.gca().invert_yaxis()


plt.tight_layout()


lime_plot_path = os.path.join(
    results_dir,
    "lime_explanation.png"
)


plt.savefig(
    lime_plot_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "LIME plot saved to:",
    lime_plot_path
)


# ============================================================
# 21. FAIRNESS AUDIT
# ============================================================

print("\n" + "=" * 60)
print("FAIRNESS AUDIT WITH FAIRLEARN")
print("=" * 60)


print(
    "\nImportant:"
)

print(
    "The dataset does not contain a direct demographic "
    "attribute such as gender or race."
)

print(
    "Therefore, country is used as a geographic grouping "
    "attribute for an exploratory fairness audit."
)


# ============================================================
# 22. CREATE COUNTRY GROUP
# ============================================================

country_series = (
    df["country"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


def create_country_group(country):

    if country == "Unknown" or country == "":

        return "Unknown"

    elif country == "United States":

        return "United States"

    else:

        return "Other"


df["country_group"] = country_series.apply(
    create_country_group
)


print("\nCountry group distribution:")

print(
    df["country_group"].value_counts()
)


# ============================================================
# 23. GET TEST SET SENSITIVE ATTRIBUTE
# ============================================================

# X_test contains the original dataframe indices.
# Use those indices to retrieve country groups.

sensitive_test = df.loc[
    X_test.index,
    "country_group"
]


print("\nTest-set country groups:")

print(
    sensitive_test.value_counts()
)


# ============================================================
# # ============================================================
# 24. CONVERT TARGET LABELS FOR FAIRLEARN
# ============================================================

print("\n" + "=" * 60)
print("FAIRLEARN LABEL ENCODING")
print("=" * 60)

# Fairlearn fairness metrics work most reliably with
# binary numerical labels.
#
# Movie   = 0
# TV Show = 1
#
# TV Show is therefore treated as the positive class.

label_mapping = {
    "Movie": 0,
    "TV Show": 1
}

y_test_fair = (
    y_test
    .map(label_mapping)
    .astype(int)
)

y_pred_fair = pd.Series(
    y_pred,
    index=y_test.index
).map(
    label_mapping
).astype(int)


print("\nFairness label mapping:")
print("Movie   = 0")
print("TV Show = 1")

print("\nActual label distribution:")
print(
    y_test_fair.value_counts()
)

print("\nPredicted label distribution:")
print(
    y_pred_fair.value_counts()
)


# ============================================================
# 25. DEMOGRAPHIC PARITY DIFFERENCE
# ============================================================

print("\n" + "=" * 60)
print("DEMOGRAPHIC PARITY DIFFERENCE")
print("=" * 60)


dp_difference = demographic_parity_difference(

    y_true=y_test_fair,

    y_pred=y_pred_fair,

    sensitive_features=sensitive_test
)


print(
    "\nDemographic Parity Difference:",
    round(
        dp_difference,
        6
    )
)


# ============================================================
# 26. EQUALIZED ODDS DIFFERENCE
# ============================================================

print("\n" + "=" * 60)
print("EQUALIZED ODDS DIFFERENCE")
print("=" * 60)


eo_difference = equalized_odds_difference(

    y_true=y_test_fair,

    y_pred=y_pred_fair,

    sensitive_features=sensitive_test
)


print(
    "\nEqualized Odds Difference:",
    round(
        eo_difference,
        6
    )
)


# ============================================================
# 27. GROUP-WISE FAIRNESS PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("GROUP-WISE PERFORMANCE")
print("=" * 60)


metric_frame = MetricFrame(

    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    },

    y_true=y_test_fair,

    y_pred=y_pred_fair,

    sensitive_features=sensitive_test
)


print("\nPerformance by country group:")

print(
    metric_frame.by_group
)


# ============================================================
# 28. SAVE FAIRNESS REPORT
# ============================================================

fairness_report = (
    metric_frame.by_group.copy()
)


fairness_report.to_csv(

    os.path.join(
        results_dir,
        "fairness_report.csv"
    )
)


print(
    "\nFairness report saved to:",
    os.path.join(
        results_dir,
        "fairness_report.csv"
    )
)


# ============================================================
# 29. FAIRNESS ACCURACY PLOT
# ============================================================

print(
    "\nGenerating fairness accuracy plot..."
)


accuracy_by_group = (
    metric_frame.by_group["accuracy"]
)


plt.figure(
    figsize=(10, 6)
)


accuracy_by_group.plot(
    kind="bar"
)


plt.xlabel(
    "Country Group"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Model Accuracy Across Country Groups"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


fairness_accuracy_path = os.path.join(
    results_dir,
    "fairness_accuracy.png"
)


plt.savefig(
    fairness_accuracy_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Fairness accuracy plot saved to:",
    fairness_accuracy_path
)


# ============================================================
# 30. FAIRNESS SELECTION RATE PLOT
# ============================================================

print(
    "\nGenerating fairness selection-rate plot..."
)


selection_by_group = (
    metric_frame.by_group["selection_rate"]
)


plt.figure(
    figsize=(10, 6)
)


selection_by_group.plot(
    kind="bar"
)


plt.xlabel(
    "Country Group"
)


plt.ylabel(
    "Selection Rate"
)


plt.title(
    "TV Show Prediction Rate Across Country Groups"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


fairness_selection_path = os.path.join(
    results_dir,
    "fairness_selection_rate.png"
)


plt.savefig(
    fairness_selection_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Fairness selection-rate plot saved to:",
    fairness_selection_path
)


# ============================================================
# 31. SAVE SHAP FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(

    os.path.join(
        results_dir,
        "shap_feature_importance.csv"
    ),

    index=False
)


# ============================================================
# 32. FAIRNESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("FAIRNESS INTERPRETATION")
print("=" * 60)


print(
    "\nDemographic Parity Difference:",
    round(dp_difference, 6)
)

print(
    "Equalized Odds Difference:",
    round(eo_difference, 6)
)


print(
    "\nValues closer to 0 indicate smaller differences "
    "between the evaluated country groups."
)

print(
    "Larger values indicate greater disparity between groups."
)

print(
    "\nNote: Country is being used only as a geographic "
    "grouping/proxy because this dataset does not contain "
    "direct demographic attributes such as gender or race."
)


# ============================================================
# 33. BIAS MITIGATION DISCUSSION
# ============================================================

print("\n" + "=" * 60)
print("POSSIBLE BIAS MITIGATION METHODS")
print("=" * 60)


print(
    "\nIf meaningful disparity is observed, possible "
    "mitigation approaches include:"
)

print(
    "1. Pre-processing: rebalance or reweight data "
    "across groups."
)

print(
    "2. In-processing: train models using fairness-aware "
    "constraints."
)

print(
    "3. Post-processing: adjust decision thresholds "
    "to reduce group disparities."
)


# ============================================================
# 34. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 5 FINAL REPORT")
print("=" * 60)


print("\nMODEL:")
print("Tuned Random Forest")


print("\nTEST ACCURACY:")
print(
    round(
        test_accuracy,
        6
    )
)


print("\nMOST IMPORTANT SHAP FEATURE:")
print(
    most_important_feature
)


print("\nLIME SAMPLE:")
print(
    "Actual:",
    actual_class
)

print(
    "Predicted:",
    predicted_class
)


print("\nFAIRNESS GROUPING:")
print(
    "United States / Other / Unknown"
)


print("\nPOSITIVE CLASS FOR FAIRNESS:")
print(
    "TV Show"
)


print("\nDEMOGRAPHIC PARITY DIFFERENCE:")
print(
    round(
        dp_difference,
        6
    )
)


print("\nEQUALIZED ODDS DIFFERENCE:")
print(
    round(
        eo_difference,
        6
    )
)


print("\nRESULT FILES:")

print("1. shap_summary.png")
print("2. shap_feature_importance.png")
print("3. shap_dependence.png")
print("4. lime_explanation.png")
print("5. lime_explanation.html")
print("6. fairness_accuracy.png")
print("7. fairness_selection_rate.png")
print("8. fairness_report.csv")
print("9. shap_feature_importance.csv")


print("\n" + "=" * 60)
print("EXPERIMENT 5 COMPLETED")
print("=" * 60)