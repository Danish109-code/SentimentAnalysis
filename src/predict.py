# ============================================================
# SENTIMENT ANALYSIS
# FINAL PREDICTION SYSTEM
# ============================================================

import re
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# 1. LOAD TF-IDF VECTORIZER
# ============================================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "models/tuned_random_forest.pkl"
)


print("=" * 60)

print("MODEL AND VECTORIZER LOADED SUCCESSFULLY!")

print("=" * 60)


# ============================================================
# 3. LOAD NLP TOOLS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# 4. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert text to lowercase

    text = text.lower()


    # Remove URLs

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )


    # Remove punctuation and numbers

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )


    # Split into words

    words = text.split()


    # Remove stopwords

    words = [
        word
        for word in words
        if word not in stop_words
    ]


    # Lemmatization

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]


    # Join words

    text = " ".join(words)


    return text


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    # Clean text

    cleaned_text = clean_text(
        text
    )


    # Convert text into TF-IDF

    text_vector = vectorizer.transform(
        [cleaned_text]
    )


    # Predict sentiment

    prediction = model.predict(
        text_vector
    )[0]


    return prediction


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

print("\n")

text = input(
    "Enter a sentence: "
)


# ============================================================
# 7. PREDICT
# ============================================================

sentiment = predict_sentiment(
    text
)


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)

print("SENTIMENT PREDICTION")

print("=" * 60)

print(
    "Input:",
    text
)

print(
    "Predicted Sentiment:",
    sentiment
)

print("=" * 60)