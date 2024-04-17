import streamlit as st
from transformers import pipeline
import requests
from PIL import Image
import os
import openai
from dotenv import load_dotenv
import io


# Load environment variables from apikey.env file
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")


# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# DALL-E endpoint for image generation
dalle_endpoint = "https://api.openai.com/v1/davinci-codex/completions"


# Main function to define the Streamlit app
def main():
    # Set the title of the app
    st.title('Tweet Sentiment Analysis')


    user_api_key = st.text_input('Enter your openAI API KEY:', type ='password')
    
    # Add a text input field for user to enter tweet
    tweet = st.text_area('Enter your tweet:', max_chars=280)

    # Create a button to trigger sentiment analysis and image generation
    if st.button('Analyze Sentiment & Generate Image'):
        # Perform sentiment analysis using the input tweet
        if tweet:
            # Make prediction using the sentiment analysis pipeline
            result = sentiment_pipeline(tweet)

            # Display the sentiment prediction
            st.write('Sentiment:', result[0]['label'], '(Confidence:', result[0]['score'], ')')

            # Generate an image based on the content of the tweet
            generated_image = generate_image(tweet,user_api_key)

            # Display the generated image
            st.image(generated_image, caption='Generated Image', use_column_width=True)
        else:
            st.write('Please enter a tweet.')
            
            
            

# Function to generate image using DALL-E
def generate_image(text, user_api_key):
    # Prepare request data
    data = {
        "prompt": text,
        "max_tokens": 50  # Adjust the max_tokens parameter as needed
    }


    # Send request to DALL-E endpoint
    response = requests.post(dalle_endpoint, json=data, headers={"Authorization": f"Bearer {user_api_key}"})

    # Extract image data from response
    image_data = response.json()["choices"][0]["text"]

    # Convert image data to PIL image
    image = Image.open(BytesIO(image_data.encode()))

    return image


# Call the main function to run the Streamlit app
if __name__ == '__main__':
    main()
