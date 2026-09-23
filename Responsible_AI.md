# Responsible AI Report

## 1. Project Overview

This project develops a machine learning system for predicting
whether a Netflix title is a Movie or TV Show using numerical
features from the merged Netflix and IMDb dataset.

The final model is a tuned Random Forest classifier.

The system uses Explainable AI techniques and fairness analysis
to improve transparency and responsible use of the model.

---

## 2. Model Used

The final selected model is a tuned Random Forest classifier.

The input features are:

- release_year
- runtimeMinutes
- averageRating
- numVotes

The final test accuracy was approximately 79.40%.

---

## 3. Explainability

SHAP and LIME were used to explain model predictions.

### SHAP

SHAP was used to understand the contribution of features to
model predictions.

The mean absolute SHAP values were:

| Feature | Mean Absolute SHAP |
|---|---:|
| runtimeMinutes | 0.174687 |
| numVotes | 0.075936 |
| release_year | 0.038698 |
| averageRating | 0.034275 |

runtimeMinutes had the highest average contribution magnitude
among the evaluated features.

### LIME

LIME was used to explain an individual prediction locally.

For the evaluated sample, the model predicted Movie while the
actual class was TV Show.

The LIME contributions showed how the local feature conditions
supported or opposed the prediction.

---

## 4. Fairness

The dataset does not contain direct demographic attributes such
as gender or race.

Therefore, country was used only as a geographic proxy for an
exploratory fairness audit.

It should not be interpreted as a direct protected demographic
attribute.

The measured fairness metrics were:

- Demographic Parity Difference: 0.140022
- Equalized Odds Difference: 0.085284

These values indicate measurable differences between the evaluated
groups. The metrics should be interpreted in the context of the
dataset and the use of country as a proxy.

---

## 5. Privacy

The project uses Netflix and IMDb dataset information for analysis
and prediction.

No personally identifiable user information is required for the
prediction API.

The dashboard only accepts the numerical model input features
required for prediction.

---

## 6. Consent

The system is intended for dataset-based analysis and demonstration.

No personal user information is collected by the dashboard for
the prediction task.

If the system is extended to collect personal information,
appropriate consent and data-use policies should be implemented.

---

## 7. Data Drift

A basic exploratory drift check was included in the dashboard.

The check compares the mean values of numerical features between
two portions of the dataset.

The features checked are:

- release_year
- runtimeMinutes
- averageRating
- numVotes

A potential drift flag indicates that the feature distribution
should be investigated further. It does not by itself establish
that production data drift is occurring.

---

## 8. Human Oversight

The model prediction should be treated as an automated prediction
rather than an absolute determination.

Users should be able to inspect the input values and model
explanations.

Model performance and fairness should be reassessed if the dataset,
features, or application purpose changes.

---

## 9. Responsible AI Checklist

| Area | Status | Description |
|---|---|---|
| Explainability | Completed | SHAP and LIME explanations implemented |
| Fairness | Completed | Fairlearn metrics evaluated |
| Privacy | Considered | No personal information required |
| Consent | Considered | No personal information collected |
| Data Drift | Implemented | Exploratory numerical drift check |
| Human Oversight | Considered | Predictions should not be treated as absolute decisions |
| Model Monitoring | Recommended | Re-evaluate performance when data changes |

---

## 10. Limitations

The model is trained on historical Netflix and IMDb data.

The fairness analysis uses country as a geographic proxy because
direct demographic attributes are not available.

The drift check is a basic exploratory check and is not a
production monitoring system.

The model should therefore be interpreted within the scope of the
available dataset and project objectives.

---

## 11. Conclusion

The project combines machine learning, explainable AI, fairness
analysis, API deployment, containerization, CI/CD and dashboard
visualization.

The Streamlit dashboard provides an interactive interface for
predictions, model metrics, SHAP explanations, fairness analysis
and data drift checks.

The Responsible AI review documents fairness, privacy, consent,
explainability and monitoring considerations.