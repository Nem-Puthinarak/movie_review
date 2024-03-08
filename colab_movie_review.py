from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Load the dataset
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            df = pd.read_csv(uploaded_file)
            
            # Perform sentiment analysis
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
            
            # Get sentiment counts
            sentiment_counts = df_subset['Sentiment'].value_counts()
            pie_chart = sentiment_counts.plot(kind='pie', labels=sentiment_counts.index, startangle=90, autopct='%1.1f%%')
            pie_chart.set_title('Sentiment Analysis Results')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.savefig('static/sentiment_analysis_result.png')
            
            return render_template('result.html', sentiment_image='static/sentiment_analysis_result.png')
        else:
            return render_template('index.html', error_message="Please upload a file.")

if __name__ == "__main__":
    app.run(debug=True)
