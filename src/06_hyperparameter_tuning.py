# ============================================================
# SENTIMENT ANALYSIS
# HYPERPARAMETER TUNING
# ============================================================

import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = joblib.load(
    "models/X_train_tfidf.pkl"
)

X_test = joblib.load(
    "models/X_test_tfidf.pkl"
)

y_train = joblib.load(
    "models/y_train.pkl"
)

y_test = joblib.load(
    "models/y_test.pkl"
)


print("=" * 60)

print("DATA LOADED SUCCESSFULLY!")

print("=" * 60)

print(
    "Training Shape:",
    X_train.shape
)

print(
    "Testing Shape:",
    X_test.shape
)


# ============================================================
# 2. CREATE RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 3. DEFINE PARAMETERS
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
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


# ============================================================
# 4. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=cv,

    scoring="f1_weighted",

    n_jobs=-1,

    verbose=2
)


print("\n" + "=" * 60)

print("STARTING HYPERPARAMETER TUNING")

print("=" * 60)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 6. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 60)

print("BEST PARAMETERS")

print("=" * 60)

print(
    grid_search.best_params_
)


# ============================================================
# 7. BEST CROSS VALIDATION SCORE
# ============================================================

print("\n" + "=" * 60)

print("BEST CROSS-VALIDATION SCORE")

print("=" * 60)

print(
    "F1 Score:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 8. GET BEST MODEL
# ============================================================

best_model = (
    grid_search.best_estimator_
)


# ============================================================
# 9. PREDICT TEST DATA
# ============================================================

y_pred = best_model.predict(
    X_test
)


# ============================================================
# 10. TEST ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("TUNED MODEL TEST ACCURACY")

print("=" * 60)

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. SAVE TUNED MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/tuned_random_forest.pkl"
)


print("\n" + "=" * 60)

print(
    "TUNED MODEL SAVED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "models/tuned_random_forest.pkl"
)