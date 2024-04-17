import streamlit as st
from transformers import pipeline
import requests
from PIL import Image
import os
from dotenv import load_dotenv
from io import BytesIO  # Import BytesIO for handling image data conversion

# Load environment variables from .env file
load_dotenv()

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# DALL-E endpoint for image generation
dalle_endpoint = "https://api.openai.com/v1/davinci-codex/completions"

# Main function to define the Streamlit app
def main():
    # Set the title of the app
    st.title('Tweet Sentiment Analysis')

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
            try:
                generated_image = generate_image(tweet)
                # Display the generated image
                st.image(generated_image, caption='Generated Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error generating image: {e}")
        else:
            st.write('Please enter a tweet.')

# Function to generate image using DALL-E
def generate_image(text):
    # Prepare request data
    data = {
        "prompt": text,
        "max_tokens": 50  # Adjust the max_tokens parameter as needed
    }

    # Get OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    # Send request to DALL-E endpoint
    response = requests.post(dalle_endpoint, json=data, headers={"Authorization": f"Bearer {api_key}"})

    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"Error: Failed to generate image. Status code: {response.status_code}")

    # Extract image data from response
    choices = response.json().get("choices", [])
    if not choices:
        raise Exception("Error: 'choices' not found in response.")
    
    image_data = choices[0].get("text")
    if not image_data:
        raise Exception("Error: Image data not found in response.")

    # Convert image data to PIL image
    image = Image.open(BytesIO(image_data.encode()))

    return image


# Call the main function to run the Streamlit app
if __name__ == '__main__':
    main()
