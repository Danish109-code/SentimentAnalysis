import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ============================================================
# SENTIMENT ANALYSIS
# TEXT PREPROCESSING
# ============================================================

import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "dataset/sentiment.csv"
)

print("Dataset Loaded Successfully!")

print(
    "Original Dataset Shape:",
    df.shape
)


# ============================================================
# 2. SELECT REQUIRED COLUMNS
# ============================================================

df = df[
    ["Text", "Sentiment"]
]
df["Sentiment"] = df["Sentiment"].astype(str).str.strip()


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=["Text", "Sentiment"]
)


print(
    "\nAfter Removing Missing Values:"
)

print(
    df.shape
)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates(
    subset=["Text"]
)


print(
    "\nAfter Removing Duplicates:"
)

print(
    df.shape
)


# ============================================================
# 5. CONVERT TEXT TO STRING
# ============================================================

df["Text"] = df["Text"].astype(
    str
)


# ============================================================
# 6. LOAD STOPWORDS
# ============================================================

stop_words = set(
    stopwords.words("english")
)


# ============================================================
# 7. CREATE LEMMATIZER
# ============================================================

lemmatizer = WordNetLemmatizer()


# ============================================================
# 8. TEXT CLEANING FUNCTION
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


    # Remove HTML tags

    text = re.sub(
        r"<.*?>",
        "",
        text
    )


    # Remove punctuation

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

    text = " ".join(
        words
    )


    return text


# ============================================================
# 9. APPLY TEXT CLEANING
# ============================================================

df["Clean_Text"] = df["Text"].apply(
    clean_text
)
# ============================================================
# CONVERT EMOTIONS INTO 3 SENTIMENT CLASSES
# ============================================================

positive_emotions = {
    "positive",
    "joy",
    "excitement",
    "contentment",
    "happy",
    "hopeful",
    "gratitude",
    "elation",
    "playful",
    "serenity",
    "empowerment",
    "enthusiasm",
    "acceptance",
    "determination",
    "inspired",
    "euphoria",
    "hope",
    "tenderness",
    "proud",
    "grateful",
    "compassionate",
    "serenity",
    "inspiration",
    "awe",
    "calmness",
    "kind",
    "pride",
    "compassion",
    "empathetic",
    "free-spirited",
    "confident",
    "accomplishment",
    "adventure",
    "surprise",
    "happiness",
    "love",
    "amusement",
    "enjoyment",
    "admiration",
    "affection",
    "adoration",
    "anticipation",
    "fulfillment",
    "reverence",
    "zest",
    "determination",
    "enchantment",
    "whimsy",
    "rejuvenation",
    "coziness",
    "thrill",
    "exploration",
    "captivation",
    "tranquility",
    "creativity",
    "satisfaction",
    "overjoyed",
    "joyfulreunion",
    "blessed",
    "appreciation",
    "confidence",
    "wonderment",
    "optimism",
    "intrigue",
    "playfuljoy",
    "mindfulness",
    "dreamchaser",
    "elegance",
    "harmony",
    "radiance",
    "wonder",
    "melodic",
    "festivejoy",
    "innerjourney",
    "freedom",
    "dazzle",
    "adrenaline",
    "artisticburst",
    "culinaryodyssey",
    "resilience",
    "immersion",
    "spark",
    "marvel",
    "success",
    "amazement",
    "romance",
    "grandeur",
    "energy",
    "celebration",
    "charm",
    "ecstasy",
    "colorful",
    "hypnotic",
    "connection",
    "iconic",
    "engagement",
    "touched",
    "triumph",
    "heartwarming",
    "sympathy",
    "renewedeffort",
    "breakthrough",
    "solace",
    "joyinbaking",
    "envisioninghistory",
    "imagination",
    "vibrancy",
    "mesmerizing",
    "culinaryadventure",
    "wintermagic",
    "thrillingjourney",
    "nature'sbeauty",
    "celestialwonder",
    "creativeinspiration",
    "runwaycreativity",
    "ocean'sfreedom",
    "whispersofthepast",
    "relief"
}


negative_emotions = {
    "negative",
    "sad",
    "despair",
    "hate",
    "bad",
    "embarrassed",
    "loneliness",
    "frustrated",
    "bitterness",
    "frustration",
    "indifference",
    "numbness",
    "melancholy",
    "desolation",
    "grief",
    "betrayal",
    "resentment",
    "boredom",
    "fearful",
    "overwhelmed",
    "jealous",
    "devastated",
    "envious",
    "dismissive",
    "anger",
    "fear",
    "sadness",
    "disgust",
    "disappointed",
    "shame",
    "jealousy",
    "envy",
    "regret",
    "apprehensive",
    "isolation",
    "disappointment",
    "heartbreak",
    "loss",
    "anxiety",
    "intimidation",
    "helplessness",
    "suffering",
    "emotionalstorm",
    "lostlove",
    "exhaustion",
    "sorrow",
    "darkness",
    "desperation",
    "ruins",
    "heartache",
    "solitude",
    "pressure",
    "miscalculation",
    "obstacle",
    "challenge"
}


neutral_emotions = {
    "neutral",
    "curiosity",
    "confusion",
    "ambivalence",
    "nostalgia",
    "reflection",
    "contemplation",
    "pensive",
    "suspense",
    "emotion",
    "journey"
}


def convert_sentiment(emotion):

    emotion = str(emotion).strip().lower()

    # Remove spaces for labels such as
    # "Joyful Reunion"

    emotion_key = emotion.replace(" ", "")

    if emotion in positive_emotions:
        return "Positive"

    if emotion_key in positive_emotions:
        return "Positive"

    if emotion in negative_emotions:
        return "Negative"

    if emotion_key in negative_emotions:
        return "Negative"

    if emotion in neutral_emotions:
        return "Neutral"

    if emotion_key in neutral_emotions:
        return "Neutral"

    # Unknown emotions
    return "Neutral"


# Apply sentiment conversion

df["Sentiment"] = df["Sentiment"].apply(
    convert_sentiment
)


# Display new distribution

print("\nNew Sentiment Distribution:")

print(
    df["Sentiment"].value_counts()
)


# ============================================================
# 10. DISPLAY ORIGINAL AND CLEAN TEXT
# ============================================================

print("\nOriginal vs Clean Text:")

print(
    df[
        ["Text", "Clean_Text"]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 11. REMOVE EMPTY TEXT
# ============================================================

df = df[
    df["Clean_Text"].str.strip() != ""
]


# ============================================================
# 12. RESET INDEX
# ============================================================

df = df.reset_index(
    drop=True
)


# ============================================================
# 13. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "dataset/processed_sentiment.csv",
    index=False
)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)

print(
    "TEXT PREPROCESSING COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print(
    "Final Dataset Shape:",
    df.shape
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)

print(
    "\nProcessed Dataset Saved At:"
)

print(
    "dataset/processed_sentiment.csv"
)