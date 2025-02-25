import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER lexicon (first time only)
nltk.download("vader_lexicon")

def analyze_sentiment(text):
    # Returns the sentiment score of a given text using VADER.
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]

    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def apply_sentiment_analysis(df):
    # Applies sentiment analysis to the chat DataFrame.
    df["sentiment"] = df["message"].apply(analyze_sentiment)
    return df

def visualize_sentiment_distribution(df):
    # Plots the sentiment distribution as a pie chart.
    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    sentiment_counts.plot.pie(autopct="%1.1f%%", colors=["green", "gray", "red"], ax=ax)
    ax.set_ylabel("")
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

def sentiment_over_time(df, st):
    # Plots the sentiment trend over time.
    df["only_date"] = pd.to_datetime(df["only_date"])  # Convert to datetime
    sentiment_counts = df.groupby("only_date")["sentiment"].value_counts().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sentiment_counts.plot(ax=ax, marker="o", linestyle="-")
    
    ax.set_title("Sentiment Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Message Count")
    ax.legend(title="Sentiment")
    plt.xticks(rotation=45)

    st.pyplot(fig)
