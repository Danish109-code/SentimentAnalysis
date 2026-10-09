# ============================================================
# SENTIMENT ANALYSIS WEB APPLICATION
# ============================================================

import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# 2. LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/tuned_random_forest.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# 3. LOAD NLP TOOLS
# ============================================================

@st.cache_resource
def download_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

download_nltk_resources()


stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# 4. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to lowercase

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


    # Split text into words

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

    return " ".join(words)


# ============================================================
# 5. SENTIMENT PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    # Clean text

    cleaned_text = clean_text(
        text
    )


    # Convert text to TF-IDF

    text_vector = vectorizer.transform(
        [cleaned_text]
    )


    # Predict sentiment

    prediction = model.predict(
        text_vector
    )[0]


    return prediction


# ============================================================
# 6. APPLICATION TITLE
# ============================================================

st.title(
    "💬 Sentiment Analysis"
)

st.write(
    "Machine Learning based Sentiment Analysis"
)


# ============================================================
# 7. USER INPUT
# ============================================================

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really enjoyed this product!",
    height=150
)


# ============================================================
# 8. ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if text.strip() == "":

        st.warning(
            "Please enter some text first."
        )

    else:

        sentiment = predict_sentiment(
            text
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader(
            "Prediction"
        )


        if sentiment == "Positive":

            st.success(
                "😊 Positive Sentiment"
            )


        elif sentiment == "Negative":

            st.error(
                "😞 Negative Sentiment"
            )


        elif sentiment == "Neutral":

            st.info(
                "😐 Neutral Sentiment"
            )


        else:

            st.write(
                f"Sentiment: {sentiment}"
            )


# ============================================================
# 9. SIDEBAR
# ============================================================

st.sidebar.title(
    "About the Project"
)

st.sidebar.write(
    """
    This application uses:

    • Text preprocessing
    • TF-IDF Vectorization
    • Random Forest
    • Machine Learning

    The model predicts:

    • Positive
    • Negative
    • Neutral
    """
)


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Sentiment Analysis | Machine Learning Project"
)
