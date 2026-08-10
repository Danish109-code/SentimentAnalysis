# ============================================================
# SENTIMENT ANALYSIS
# MODEL EVALUATION
# ============================================================

import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD BEST MODEL
# ============================================================

model = joblib.load(
    "models/best_sentiment_model.pkl"
)

print("=" * 60)

print("BEST MODEL LOADED SUCCESSFULLY!")

print("=" * 60)


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

X_test = joblib.load(
    "models/X_test_tfidf.pkl"
)

y_test = joblib.load(
    "models/y_test.pkl"
)


print("\nTest Data Loaded Successfully!")

print(
    "Test Data Shape:",
    X_test.shape
)


# ============================================================
# 3. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


print("\nPredictions Generated Successfully!")


# ============================================================
# 4. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("MODEL ACCURACY")

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
# 5. CLASSIFICATION REPORT
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
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

print(cm)


# ============================================================
# 7. GET CLASS NAMES
# ============================================================

class_names = sorted(
    y_test.unique()
)


print("\nClass Names:")

print(class_names)


# ============================================================
# 8. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 9. CONFUSION MATRIX HEATMAP
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Sentiment Analysis - Confusion Matrix"
)

plt.xlabel(
    "Predicted Sentiment"
)

plt.ylabel(
    "Actual Sentiment"
)

plt.tight_layout()


plt.savefig(
    "models/confusion_matrix.png",
    dpi=300
)


plt.show()


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

results_df = joblib.load(
    "models/y_test.pkl"
)

results_df = results_df.reset_index(
    drop=True
)

predictions_df = results_df.to_frame(
    name="Actual"
)

predictions_df["Predicted"] = y_pred


predictions_df.to_csv(
    "models/predictions.csv",
    index=False
)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL EVALUATION COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "\nSaved:"
)

print(
    "models/confusion_matrix.png"
)

print(
    "models/predictions.csv"
)