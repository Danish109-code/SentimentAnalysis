# ============================================================
# SENTIMENT ANALYSIS
# MACHINE LEARNING MODEL TRAINING
# ============================================================

import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD TF-IDF DATA
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

print("TF-IDF DATA LOADED SUCCESSFULLY!")

print("=" * 60)

print(
    "Training Data Shape:",
    X_train.shape
)

print(
    "Testing Data Shape:",
    X_test.shape
)


# ============================================================
# 2. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
# 3. CREATE RESULTS DICTIONARY
# ============================================================

results = {}


# ============================================================
# 4. TRAIN MODELS
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 60)

    print(
        "Training:",
        name
    )

    print("=" * 60)


    # Train model

    model.fit(
        X_train,
        y_train
    )


    # Make predictions

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # Calculate Metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # Store Results
    # ========================================================

    results[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    }


    # ========================================================
    # Display Results
    # ========================================================

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )


    # ========================================================
    # Save Model
    # ========================================================

    filename = name.lower().replace(
        " ",
        "_"
    )

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )


# ============================================================
# 5. DISPLAY MODEL COMPARISON
# ============================================================

print("\n")

print("=" * 80)

print("MODEL COMPARISON")

print("=" * 80)


for name, metrics in results.items():

    print("\nModel:", name)

    print(
        "Accuracy :",
        round(
            metrics["Accuracy"],
            4
        )
    )

    print(
        "Precision:",
        round(
            metrics["Precision"],
            4
        )
    )

    print(
        "Recall   :",
        round(
            metrics["Recall"],
            4
        )
    )

    print(
        "F1 Score :",
        round(
            metrics["F1 Score"],
            4
        )
    )


# ============================================================
# 6. FIND BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda x: results[x]["F1 Score"]
)


best_model_score = results[
    best_model_name
]["F1 Score"]


print("\n")

print("=" * 60)

print("BEST MODEL")

print("=" * 60)

print(
    "Model:",
    best_model_name
)

print(
    "F1 Score:",
    round(
        best_model_score,
        4
    )
)


# ============================================================
# 7. SAVE BEST MODEL
# ============================================================

best_model_filename = (
    best_model_name
    .lower()
    .replace(" ", "_")
)

best_model = models[
    best_model_name
]

joblib.dump(
    best_model,
    "models/best_sentiment_model.pkl"
)


print("\nBest model saved successfully!")

print(
    "models/best_sentiment_model.pkl"
)


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY!"
)

print("=" * 60)