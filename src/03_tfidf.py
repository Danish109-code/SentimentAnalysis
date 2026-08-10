# ============================================================
# SENTIMENT ANALYSIS
# TRAIN/TEST SPLIT + TF-IDF
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
import os


# ============================================================
# 1. LOAD PROCESSED DATASET
# ============================================================

df = pd.read_csv(
    "dataset/processed_sentiment.csv"
)

print("Processed Dataset Loaded Successfully!")

print(
    "Dataset Shape:",
    df.shape
)


# ============================================================
# 2. DISPLAY COLUMNS
# ============================================================

print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# 3. CHECK SENTIMENT DISTRIBUTION
# ============================================================

print("\nSentiment Distribution:")

print(
    df["Sentiment"].value_counts()
)


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================

X = df["Clean_Text"]

y = df["Sentiment"]


# ============================================================
# 5. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. DISPLAY SPLIT INFORMATION
# ============================================================

print("\nTrain/Test Split Completed!")

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================================
# 7. CREATE TF-IDF VECTORIZER
# ============================================================

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


# ============================================================
# 8. FIT TF-IDF ON TRAINING DATA
# ============================================================

X_train_tfidf = tfidf.fit_transform(
    X_train
)


# ============================================================
# 9. TRANSFORM TEST DATA
# ============================================================

X_test_tfidf = tfidf.transform(
    X_test
)


# ============================================================
# 10. DISPLAY TF-IDF SHAPES
# ============================================================

print("\nTF-IDF Transformation Completed!")

print(
    "X_train TF-IDF Shape:",
    X_train_tfidf.shape
)

print(
    "X_test TF-IDF Shape:",
    X_test_tfidf.shape
)


# ============================================================
# 11. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 12. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# 13. SAVE TRAIN/TEST DATA
# ============================================================

joblib.dump(
    X_train_tfidf,
    "models/X_train_tfidf.pkl"
)

joblib.dump(
    X_test_tfidf,
    "models/X_test_tfidf.pkl"
)

joblib.dump(
    y_train,
    "models/y_train.pkl"
)

joblib.dump(
    y_test,
    "models/y_test.pkl"
)


# ============================================================
# 14. SAVE SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 60)

print(
    "TF-IDF PROCESSING COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "\nSaved files:"
)

print(
    "models/tfidf_vectorizer.pkl"
)

print(
    "models/X_train_tfidf.pkl"
)

print(
    "models/X_test_tfidf.pkl"
)

print(
    "models/y_train.pkl"
)

print(
    "models/y_test.pkl"
)