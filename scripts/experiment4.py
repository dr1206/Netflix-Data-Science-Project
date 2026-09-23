import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/cleaned/merged_v1.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "release_year",
    "runtimeMinutes",
    "averageRating",
    "numVotes"
]

target = "type"

X = df[features]
y = df[target]

print("\nFeatures used:")
print(features)

print("\nTarget variable:", target)

print("\nTarget classes:")
print(y.unique())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nDataset preparation completed.")


# ============================================================
# 4. FEATURE SCALING
# ============================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])


# ============================================================
# 5. DEFINE BASELINE MODELS
# ============================================================

models = {

    "Logistic Regression": logistic_model,

    "KNN": knn_model,

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": svm_model
}


# ============================================================
# 6. BASELINE MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL TRAINING")
print("=" * 60)

results = []

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Store trained model
    trained_models[name] = model

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(
        "Accuracy:",
        round(accuracy, 6)
    )


# ============================================================
# 7. BASELINE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.6f}".format
        }
    )
)


# ============================================================
# 8. BEST BASELINE MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_accuracy = results_df.iloc[0]["Accuracy"]


print("\n" + "=" * 60)
print("BEST BASELINE MODEL")
print("=" * 60)

print("Model:", best_model_name)

print(
    "Accuracy:",
    round(best_accuracy, 4)
)

print("\nBaseline model training completed.")


# ============================================================
# 9. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

print("\nThe two best baseline models will be tuned:")

print("1. Random Forest")
print("2. KNN")


# ============================================================
# 10. GET BASELINE ACCURACIES
# ============================================================

rf_baseline_accuracy = results_df.loc[
    results_df["Model"] == "Random Forest",
    "Accuracy"
].values[0]

knn_baseline_accuracy = results_df.loc[
    results_df["Model"] == "KNN",
    "Accuracy"
].values[0]


# ============================================================
# 11. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

rf = RandomForestClassifier(
    random_state=42
)


rf_param_grid = {

    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        None,
        10,
        20
    ],

    "min_samples_split": [
        2,
        5
    ],

    "min_samples_leaf": [
        1,
        2
    ]
}


print("\nTesting Random Forest parameter combinations...")


rf_grid = GridSearchCV(

    estimator=rf,

    param_grid=rf_param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=1
)


rf_grid.fit(
    X_train,
    y_train
)


print("\nBest Random Forest parameters:")

print(
    rf_grid.best_params_
)


print("\nBest Random Forest cross-validation accuracy:")

print(
    round(
        rf_grid.best_score_,
        4
    )
)


# ============================================================
# 12. EVALUATE TUNED RANDOM FOREST
# ============================================================

best_rf = rf_grid.best_estimator_


tuned_rf_pred = best_rf.predict(
    X_test
)


rf_tuned_accuracy = accuracy_score(
    y_test,
    tuned_rf_pred
)


print("\nTuned Random Forest test accuracy:")

print(
    round(
        rf_tuned_accuracy,
        6
    )
)


# ============================================================
# 13. KNN HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("KNN HYPERPARAMETER TUNING")
print("=" * 60)


knn_pipeline = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        KNeighborsClassifier()
    )
])


knn_param_grid = {

    "model__n_neighbors": [
        3,
        5,
        7,
        9,
        11
    ],

    "model__weights": [
        "uniform",
        "distance"
    ],

    "model__p": [
        1,
        2
    ]
}


print("\nTesting KNN parameter combinations...")


knn_grid = GridSearchCV(

    estimator=knn_pipeline,

    param_grid=knn_param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=1
)


knn_grid.fit(
    X_train,
    y_train
)


print("\nBest KNN parameters:")

print(
    knn_grid.best_params_
)


print("\nBest KNN cross-validation accuracy:")

print(
    round(
        knn_grid.best_score_,
        4
    )
)


# ============================================================
# 14. EVALUATE TUNED KNN
# ============================================================

best_knn = knn_grid.best_estimator_


tuned_knn_pred = best_knn.predict(
    X_test
)


knn_tuned_accuracy = accuracy_score(
    y_test,
    tuned_knn_pred
)


print("\nTuned KNN test accuracy:")

print(
    round(
        knn_tuned_accuracy,
        6
    )
)


# ============================================================
# 15. BASELINE VS TUNED COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Random Forest",
        "Random Forest",
        "KNN",
        "KNN"
    ],

    "Version": [
        "Baseline",
        "Tuned",
        "Baseline",
        "Tuned"
    ],

    "Accuracy": [
        rf_baseline_accuracy,
        rf_tuned_accuracy,
        knn_baseline_accuracy,
        knn_tuned_accuracy
    ]
})


print("\n" + "=" * 60)
print("BASELINE VS TUNED COMPARISON")
print("=" * 60)


print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.6f}".format
        }
    )
)


# ============================================================
# 16. ACCURACY IMPROVEMENT
# ============================================================

rf_improvement = (
    rf_tuned_accuracy -
    rf_baseline_accuracy
)


knn_improvement = (
    knn_tuned_accuracy -
    knn_baseline_accuracy
)


print("\n" + "=" * 60)
print("ACCURACY IMPROVEMENT")
print("=" * 60)


print(
    "Random Forest improvement:",
    round(
        rf_improvement,
        6
    )
)


print(
    "KNN improvement:",
    round(
        knn_improvement,
        6
    )
)


# ============================================================
# 17. SELECT BEST TUNED MODEL
# ============================================================

if rf_tuned_accuracy >= knn_tuned_accuracy:

    final_model = best_rf

    final_model_name = "Tuned Random Forest"

    final_accuracy = rf_tuned_accuracy

    final_params = rf_grid.best_params_

else:

    final_model = best_knn

    final_model_name = "Tuned KNN"

    final_accuracy = knn_tuned_accuracy

    final_params = knn_grid.best_params_


print("\n" + "=" * 60)
print("BEST TUNED MODEL")
print("=" * 60)


print(
    "Model:",
    final_model_name
)


print(
    "Test Accuracy:",
    round(
        final_accuracy,
        6
    )
)


# ============================================================
# 18. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 19. SAVE BEST MODEL USING JOBLIB
# ============================================================

model_path = os.path.join(
    "models",
    "best_model.pkl"
)


joblib.dump(
    final_model,
    model_path
)


print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

print(
    "Best model saved to:",
    model_path
)


# ============================================================
# 20. MLFLOW EXPERIMENT SETUP
# ============================================================

print("\n" + "=" * 60)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 60)


mlflow.set_experiment(
    "Netflix_ML_Experiment"
)


print(
    "\nMLflow experiment:",
    "Netflix_ML_Experiment"
)


# ============================================================
# 21. MLFLOW - BASELINE MODEL TRACKING
# ============================================================

print("\nTracking baseline models with MLflow...")


for name, model in trained_models.items():

    # Get accuracy
    model_accuracy = results_df.loc[
        results_df["Model"] == name,
        "Accuracy"
    ].values[0]

    with mlflow.start_run(
        run_name="Baseline_" + name.replace(" ", "_")
    ):

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            name
        )

        mlflow.log_param(
            "dataset",
            "merged_v1.csv"
        )

        mlflow.log_param(
            "features",
            ",".join(features)
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # ----------------------------------------------------
        # Log metric
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            float(model_accuracy)
        )

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        model_params = model.get_params()

        for param_name, param_value in model_params.items():

            mlflow.log_param(
                "model_" + param_name,
                str(param_value)
            )

        # ----------------------------------------------------
        # Log sklearn model
        #
        # IMPORTANT:
        # pickle avoids the KNN skops trusted-types error
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle"
        )

        print(
            "Logged:",
            name,
            "| Accuracy:",
            round(
                model_accuracy,
                6
            )
        )


# ============================================================
# 22. MLFLOW - TUNED RANDOM FOREST
# ============================================================

print("\nTracking tuned Random Forest...")


with mlflow.start_run(
    run_name="Tuned_Random_Forest"
):

    # Log model type
    mlflow.log_param(
        "model_type",
        "Random Forest"
    )

    # Log tuning method
    mlflow.log_param(
        "tuning_method",
        "GridSearchCV"
    )

    # Log CV folds
    mlflow.log_param(
        "cv_folds",
        5
    )

    # Log features
    mlflow.log_param(
        "features",
        ",".join(features)
    )

    # Log dataset
    mlflow.log_param(
        "dataset",
        "merged_v1.csv"
    )

    # Log best parameters
    for param_name, param_value in rf_grid.best_params_.items():

        mlflow.log_param(
            "best_" + param_name,
            str(param_value)
        )

    # Log metrics
    mlflow.log_metric(
        "cv_accuracy",
        float(rf_grid.best_score_)
    )

    mlflow.log_metric(
        "test_accuracy",
        float(rf_tuned_accuracy)
    )

    mlflow.log_metric(
        "baseline_accuracy",
        float(rf_baseline_accuracy)
    )

    mlflow.log_metric(
        "accuracy_improvement",
        float(rf_improvement)
    )

    # Log model using pickle
    mlflow.sklearn.log_model(
        best_rf,
        name="tuned_random_forest",
        serialization_format="pickle"
    )

    print(
        "Tuned Random Forest logged successfully."
    )


# ============================================================
# 23. MLFLOW - TUNED KNN
# ============================================================

print("\nTracking tuned KNN...")


with mlflow.start_run(
    run_name="Tuned_KNN"
):

    # Log model type
    mlflow.log_param(
        "model_type",
        "KNN"
    )

    # Log tuning method
    mlflow.log_param(
        "tuning_method",
        "GridSearchCV"
    )

    # Log CV folds
    mlflow.log_param(
        "cv_folds",
        5
    )

    # Log features
    mlflow.log_param(
        "features",
        ",".join(features)
    )

    # Log dataset
    mlflow.log_param(
        "dataset",
        "merged_v1.csv"
    )

    # Log best parameters
    for param_name, param_value in knn_grid.best_params_.items():

        mlflow.log_param(
            "best_" + param_name.replace(
                "model__",
                ""
            ),
            str(param_value)
        )

    # Log metrics
    mlflow.log_metric(
        "cv_accuracy",
        float(knn_grid.best_score_)
    )

    mlflow.log_metric(
        "test_accuracy",
        float(knn_tuned_accuracy)
    )

    mlflow.log_metric(
        "baseline_accuracy",
        float(knn_baseline_accuracy)
    )

    mlflow.log_metric(
        "accuracy_improvement",
        float(knn_improvement)
    )

    # Log model using pickle
    mlflow.sklearn.log_model(
        best_knn,
        name="tuned_knn",
        serialization_format="pickle"
    )

    print(
        "Tuned KNN logged successfully."
    )


# ============================================================
# 24. MLFLOW - FINAL BEST MODEL
# ============================================================

print("\nTracking final best model...")


with mlflow.start_run(
    run_name="FINAL_BEST_MODEL"
):

    # --------------------------------------------------------
    # Log parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "model_type",
        final_model_name
    )

    mlflow.log_param(
        "dataset",
        "merged_v1.csv"
    )

    mlflow.log_param(
        "features",
        ",".join(features)
    )

    mlflow.log_param(
        "selection_method",
        "Highest test accuracy"
    )

    # --------------------------------------------------------
    # Log final accuracy
    # --------------------------------------------------------

    mlflow.log_metric(
        "test_accuracy",
        float(final_accuracy)
    )

    # --------------------------------------------------------
    # Log final model parameters
    # --------------------------------------------------------

    for param_name, param_value in final_params.items():

        mlflow.log_param(
            "best_" + param_name,
            str(param_value)
        )

    # --------------------------------------------------------
    # Log Joblib model artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        model_path
    )

    # --------------------------------------------------------
    # Log model using pickle
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        final_model,
        name="final_best_model",
        serialization_format="pickle"
    )

    print(
        "Final best model logged successfully."
    )


# ============================================================
# 25. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 4 COMPLETED")
print("=" * 60)

print(
    "\nBest baseline model:",
    best_model_name
)

print(
    "Best baseline accuracy:",
    round(
        best_accuracy,
        4
    )
)

print(
    "\nFinal best model:",
    final_model_name
)

print(
    "Final test accuracy:",
    round(
        final_accuracy,
        4
    )
)

print(
    "\nModel file:",
    model_path
)

print(
    "\nMLflow experiment:",
    "Netflix_ML_Experiment"
)

print(
    "\nMLflow tracking completed successfully."
)

print("=" * 60)