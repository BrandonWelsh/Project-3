import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Main function to define the Streamlit app
def main():
    # Set the title of the app
    st.title('Tweet Sentiment Analysis')

    # Add a text input field for user to enter tweet
    tweet = st.text_area('Enter your tweet:', max_chars=280)

    # Create a button to trigger sentiment analysis
    if st.button('Analyze Sentiment'):
        # Perform sentiment analysis using the input tweet
        if tweet:
            # Make prediction using the sentiment analysis pipeline
            result = sentiment_pipeline(tweet)

            # Display the sentiment prediction
            st.write('Sentiment:', result[0]['label'], '(Confidence:', result[0]['score'], ')')
        else:
            st.write('Please enter a tweet.')

# Call the main function to run the Streamlit app
if __name__ == '__main__':
    main()
