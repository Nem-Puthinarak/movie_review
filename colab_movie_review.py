import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def perform_sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()
    df_subset = df.head(500)

    def assign_sentiment(row):
        score = sid.polarity_scores(row)['compound']
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df_subset['Sentiment'] = df_subset['text'].apply(assign_sentiment)
    return df_subset['Sentiment'].value_counts()

def main():
    st.title("Movie Reviews Intelligent Analysis System")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Sample of the dataset:")
        st.write(df.head())

        st.write("### Sentiment Analysis Results:")
        sentiment_counts = perform_sentiment_analysis(df)
        st.write(sentiment_counts)

        # Plot pie chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, startangle=90, autopct='%1.1f%%')
        ax.set_title('Sentiment Analysis Results')
        st.pyplot(fig)

if __name__ == "__main__":
    main()

