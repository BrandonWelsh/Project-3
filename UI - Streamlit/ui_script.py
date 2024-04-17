import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate image from text description
def generate_image(text_description):
    # Encode the text description
    inputs = processor(text_description, return_tensors="pt", padding=True)
    
    # Generate image features from text using CLIP model
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    # Decode image features to obtain the generated image
    generated_image = processor.decode(image_features)
    
    return generated_image

# Streamlit app
def main():
    # Set the title of the app
    st.title('Image Generation from Text')

    # Add a text input field for user to enter text description
    text_description = st.text_input('Enter text description:', 'a cat sitting on a table')

    # Create a button to trigger image generation
    if st.button('Generate Image'):
        # Generate image from text description
        generated_image = generate_image(text_description)
        
        # Display the generated image
        st.image(generated_image, caption='Generated Image', use_column_width=True)

# Call the main function to run the Streamlit app
if __name__ ==
