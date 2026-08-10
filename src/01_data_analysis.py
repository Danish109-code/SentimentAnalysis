import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset

df = pd.read_csv(
    "dataset/sentiment.csv"
)


# Dataset shape

print("Dataset Shape:")
print(df.shape)


# First 5 rows

print("\nFirst 5 Rows:")
print(df.head())


# Column names

print("\nColumns:")
print(df.columns)


# Dataset information

print("\nDataset Information:")
print(df.info())


# Missing values

print("\nMissing Values:")
print(df.isnull().sum())


# Duplicate rows

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# Sentiment distribution

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())


# Plot sentiment distribution

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sentiment"
)

plt.title(
    "Sentiment Distribution"
)

plt.xlabel(
    "Sentiment"
)

plt.ylabel(
    "Number of Reviews"
)

plt.tight_layout()

plt.show()