import streamlit as st
from transformers import pipeline
import os
import openai
import base64
import pickle

# Load environment variables from apikey.env file
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Main function to define the Streamlit app
def main():
    st.title('Tweet Sentiment Analysis and Image Generation')

    user_api_key = st.text_input('Enter your openAI API KEY:', type='password')
    
    tweet = st.text_area('Enter your tweet:', max_chars=280)

    if st.button('Analyze Sentiment & Generate Image'):
        if tweet:
            result = sentiment_pipeline(tweet)
            st.write('Sentiment from Transformer:', result[0]['label'], 'Confidence:', result[0]['score'])
            

            #generated_prediction = generate_prediction_bert(tweet)  
            #st.write('Sentiment from Bert Model:', generated_prediction) #'Confidence:', result[0]['score'])
            
            generated_prediction_our_model = generate_prediction(tweet)  
            st.write('Sentiment from Our Model:', generated_prediction_our_model) #'Confidence:', result[0]['score'])
                
            
            # generated_image = generate_image(tweet, user_api_key)
            # if generated_image:
            #     st.image(generated_image, caption='Generated Image', use_column_width=True)
            # else:
            #     st.write("Failed to generate image. Please check the input and API settings.")
                
            
        else:
            st.write('Please enter a tweet.')

# Function to generate image using DALL-E
def generate_image(text, user_api_key):
    # Initialize OpenAI client
    openai.api_key = user_api_key

    # Make a request to DALL-E
    response = openai.Image.create(
        model="dall-e-3",
        quality = "standard", 
        prompt=text,
        n=1,
        size="1024x1024"
    )

    # Get the URL of the generated image
    image_url = response.data[0].url

    # Return the URL (you can also download the image and return it as bytes)
    return image_url



#define function to call for our model 
def generate_prediction_bert(text):

    
    
    #tokenize the tweet
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  #needs revision
    #load the model file and run the model to get the sentiment
    
    #predict from the model
    
    #return the prediction
    
    prediction = "TEST"  #get predictiopn from model
    
    return prediction






def generate_prediction(text):
    try:
        # Load the model from the pickle file
        with open('../model_directory/sentiment_model.pk1', 'rb') as f:
            model = pickle.load(f)
        
        print("Model loaded successfully")
        
        # Make predictions
        prediction = model.predict(text)  # Adjust this line based on how your model makes predictions
        
        print("Prediction:", prediction)
        
        return prediction
    except Exception as e:
        print("Error during prediction:", e)
        return None

if __name__ == '__main__':
    main()
