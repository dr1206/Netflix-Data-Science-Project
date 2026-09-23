import os

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Netflix AI Dashboard",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/cleaned/merged_v1.csv"

SHAP_SUMMARY = "results/experiment5/shap_summary.png"
SHAP_IMPORTANCE = "results/experiment5/shap_feature_importance.png"
SHAP_DEPENDENCE = "results/experiment5/shap_dependence.png"

FAIRNESS_ACCURACY = "results/experiment5/fairness_accuracy.png"
FAIRNESS_SELECTION = "results/experiment5/fairness_selection_rate.png"
FAIRNESS_REPORT = "results/experiment5/fairness_report.csv"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Netflix AI Prediction & Responsible AI Dashboard")

st.write(
    "Interactive dashboard for Netflix content prediction, "
    "model performance, explainability, fairness and data drift analysis."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Prediction",
        "Model Performance",
        "SHAP Explainability",
        "Fairness Analysis",
        "Drift Check"
    ]
)


# ==================================================
# 1. PREDICTION
# ==================================================

if page == "Prediction":

    st.header("Netflix Content Prediction")

    st.write(
        "Enter the numerical features of a Netflix title "
        "to predict whether it is a Movie or TV Show."
    )

    col1, col2 = st.columns(2)

    with col1:
        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2026,
            value=2019
        )

        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=1.0,
            max_value=600.0,
            value=98.0
        )

    with col2:
        rating = st.number_input(
            "IMDb Rating",
            min_value=0.0,
            max_value=10.0,
            value=6.4,
            step=0.1
        )

        votes = st.number_input(
            "Number of IMDb Votes",
            min_value=0,
            value=10000,
            step=100
        )

    if st.button("Predict", type="primary"):

        input_data = pd.DataFrame(
            [{
                "release_year": release_year,
                "runtimeMinutes": runtime,
                "averageRating": rating,
                "numVotes": votes
            }]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Content Type: {prediction}"
        )

        st.subheader("Input Features")

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ==================================================
# 2. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Final Model",
            "Tuned Random Forest"
        )

    with col2:
        st.metric(
            "Test Accuracy",
            "79.40%"
        )

    with col3:
        st.metric(
            "Training Dataset",
            f"{len(df):,} records"
        )

    st.subheader("Model Information")

    performance_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "SVM",
            "Decision Tree",
            "KNN",
            "Random Forest",
            "Tuned Random Forest"
        ],
        "Accuracy": [
            0.747216,
            0.760022,
            0.773942,
            0.783408,
            0.787862,
            0.793987
        ]
    })

    performance_data["Accuracy (%)"] = (
        performance_data["Accuracy"] * 100
    ).round(2)

    st.dataframe(
        performance_data[
            ["Model", "Accuracy (%)"]
        ],
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Movies",
            f"{(df['type'] == 'Movie').sum():,}"
        )

    with col3:
        st.metric(
            "TV Shows",
            f"{(df['type'] == 'TV Show').sum():,}"
        )

    with col4:
        st.metric(
            "Features",
            "4"
        )


# ==================================================
# 3. SHAP EXPLAINABILITY
# ==================================================

elif page == "SHAP Explainability":

    st.header("SHAP Explainability")

    st.write(
        "SHAP explains how features contribute to the model's "
        "predictions. The final model is a tuned Random Forest."
    )

    if os.path.exists(SHAP_SUMMARY):

        st.subheader("SHAP Summary Plot")

        st.image(
            SHAP_SUMMARY,
            use_container_width=True
        )

    if os.path.exists(SHAP_IMPORTANCE):

        st.subheader("Global Feature Importance")

        st.image(
            SHAP_IMPORTANCE,
            use_container_width=True
        )

    st.subheader("Feature Importance")

    shap_data = pd.DataFrame({
        "Feature": [
            "runtimeMinutes",
            "numVotes",
            "release_year",
            "averageRating"
        ],
        "Mean Absolute SHAP": [
            0.174687,
            0.075936,
            0.038698,
            0.034275
        ]
    })

    st.dataframe(
        shap_data,
        use_container_width=True
    )

    st.info(
        "runtimeMinutes has the highest mean absolute SHAP value "
        "among the evaluated features, indicating that it has the "
        "largest average contribution magnitude to model predictions."
    )

    if os.path.exists(SHAP_DEPENDENCE):

        st.subheader("SHAP Dependence Plot")

        st.image(
            SHAP_DEPENDENCE,
            use_container_width=True
        )


# ==================================================
# 4. FAIRNESS ANALYSIS
# ==================================================

elif page == "Fairness Analysis":

    st.header("Responsible AI - Fairness Analysis")

    st.write(
        "Fairness was evaluated using country as a geographic "
        "proxy because the dataset does not contain direct "
        "demographic attributes such as gender or race."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Demographic Parity Difference",
            "0.140022"
        )

    with col2:
        st.metric(
            "Equalized Odds Difference",
            "0.085284"
        )

    st.subheader("Fairness Metrics")

    fairness_data = pd.DataFrame({
        "Metric": [
            "Demographic Parity Difference",
            "Equalized Odds Difference"
        ],
        "Value": [
            0.140022,
            0.085284
        ]
    })

    st.dataframe(
        fairness_data,
        use_container_width=True
    )

    if os.path.exists(FAIRNESS_ACCURACY):

        st.subheader("Accuracy by Group")

        st.image(
            FAIRNESS_ACCURACY,
            use_container_width=True
        )

    if os.path.exists(FAIRNESS_SELECTION):

        st.subheader("Selection Rate by Group")

        st.image(
            FAIRNESS_SELECTION,
            use_container_width=True
        )

    if os.path.exists(FAIRNESS_REPORT):

        st.subheader("Group-wise Fairness Report")

        fairness_report = pd.read_csv(
            FAIRNESS_REPORT
        )

        st.dataframe(
            fairness_report,
            use_container_width=True
        )


# ==================================================
# 5. DRIFT CHECK
# ==================================================

elif page == "Drift Check":

    st.header("Data Drift Check")

    st.write(
        "A simple drift check compares the mean values of "
        "the first and second halves of the dataset."
    )

    numerical_features = [
        "release_year",
        "runtimeMinutes",
        "averageRating",
        "numVotes"
    ]

    midpoint = len(df) // 2

    reference_data = df.iloc[:midpoint]
    current_data = df.iloc[midpoint:]

    drift_results = []

    for feature in numerical_features:

        reference_mean = reference_data[
            feature
        ].mean()

        current_mean = current_data[
            feature
        ].mean()

        if reference_mean != 0:

            percentage_change = (
                abs(current_mean - reference_mean)
                / abs(reference_mean)
            ) * 100

        else:
            percentage_change = 0

        if percentage_change > 10:
            status = "Potential Drift"
        else:
            status = "Stable"

        drift_results.append({
            "Feature": feature,
            "Reference Mean": round(
                reference_mean, 3
            ),
            "Current Mean": round(
                current_mean, 3
            ),
            "Change (%)": round(
                percentage_change, 2
            ),
            "Status": status
        })

    drift_df = pd.DataFrame(
        drift_results
    )

    st.dataframe(
        drift_df,
        use_container_width=True
    )

    st.info(
        "The drift check is an exploratory comparison of "
        "feature means. A potential drift flag does not by "
        "itself establish a production data drift problem."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write(
    "Netflix Data Science Project"
)

st.sidebar.write(
    "Responsible AI Dashboard"
)