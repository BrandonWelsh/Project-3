# Tweet Sentiment Analysis and Image Generation
### Project 3, Group 4
- Brandon Welsh
- Joe Timmons
- Omar Hassanein
- Raymond Conley

April 2024

Due date: April 22nd, 2024

## Overview

This Streamlit application performs sentiment analysis on user-provided tweets and generates corresponding images based on the sentiment of the tweets. It utilizes OpenAI's DALL-E for image generation and Hugging Face's Transformers library for sentiment analysis. Moreover, we created an addiotnal ML model to allow the user to have more than one opinon for their tweet's sentiment analysis. The streamlit app displays both results and an image generated. 


## Program Goals
The goal of our project is to produce a neural network trained on Twitter sentiment analysis data, which is capable of reading the content of a tweet provided to it and recognize the sentiment of that tweet as being either positive, neutral, or negative. It is able to accept user input for this tweet in a Streamlit interface, and display the rating of the tweet. Additionally, work was performed to develop an AI which could read the content of the tweet and provide an emoji which is most closely related to the content of the tweet, defaulting to a simple smile, frown, or neutral face (based on the sentiment score of the tweet) if it could not determine a suitable emoji.

## Features

- Analyze sentiment of tweets using Transformers pipeline.
Analysis sentiment of tweets based on a ML model developed inhouse by our team
- Generate images based on the sentiment using OpenAI's DALL-E.
- User-friendly interface built with Streamlit.
- Secure input handling with password input for API key.
- 

## Data Source
- [Tweet data source](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)

## Technology used in the app

- [Transformers Sentiment Analysis Pipeline](https://github.com/BrandonWelsh/Project-3.git): Utilized for sentiment analysis of tweets.

- [Streamlit](https://streamlit.io/): Used for building the user interface.

- [DALL-E 3](https://openai.com/dall-e-3): OpenAI's model used for image generation.



## Dependencies/Setup Instructions
Pip install all required libraries before running the following cell:

    import pandas as pd
    import numpy as np
    import joblib
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from keras.preprocessing.sequence import pad_sequences
    from catboost import CatBoostClassifier

## Dependencies/Setup Instructions (continued)
Pip install all required libraries before running the following cell:
```
$ git clone https://github.com/BrandonWelsh/Project-3.git
$ cd yourproject
pip install streamlit as st
pip install transformers
pip install os
pip install openai
pip install base 64
```

## Usage

1. Set up your OpenAI API Key:
2. Run the Streamlit app in your browser at `http://localhost:8501 or by command prompt ```streamlit run https://raw.githubusercontent.com/BrandonWelsh/Project-3/main/UI_Streamlit/ui_script.py```

3. use your OPENAI_API_KEY on the Streamlit app
4. Enter your tweet and follow the in app instructions. 


## Team Member Responsibilities
Brandon Welsh: Performed research on the image recognition portion of the project before pivoting that over to create an emoji bot based on the sentiment of the tweet. [I will add more to this]

Joe Timmons: Preso outline & design. Readme to Preso Content Reconciliation.

Omar Hassanein:

Raymond Conley:

## Team Member Analysis
Brandon Welsh: My interpretation of the results of this project is... [2-3 paragraphs per person]

Joe Timmons: Our project successfully developed a neural network capable of analyzing the sentiment of user-inputted tweets and classifying them as positive, neutral, or negative within a Streamlit interface. While the original plan to compare text-based and image-based sentiment analysis using AI-generated visuals proved too complex given the available data and project scope, the completed sentiment analysis tool represents meaningful achievements in line with our revised project goals.

Omar Hassanein: 

Raymond Conley: 

## Resources Utilized
TODO, KEEP ADDING TO THIS

Python, Jupyter Notebooks

## Bugs
TODO, LIST BUGS AS THEY SHOW UP

## For Future Research
The emoji functionality was an add-on to the project and was done in place of the original plan. Our original plan was to take a sample (about 20) of tweets, run them through a free AI image generator (DALL-E 3 or equivalent), and then feed these images to a convolutional neural network trained on image sentiment, in order to determine the sentiment of the tweet based on the image generated by AI. We would then be able to compare the results of this model with our text-based sentiment analysis bot and see if one or both (or neither) was able to make an accurate prediction of the sentiment of the Tweet content. Unfortunately, our hubris was larger than the technology required to pull this off. The biggest problem was obtaining pre-labeled image sentiment data to use to train the CNN. I (Brandon Welsh) worked with this dataset for a while: [https://data.world/crowdflower/image-sentiment-polarity]. It contains approximately 16,000 random images which have been labeled into one of five categories: highly positive, positive, neutral, negative, and highly negative. The biggest problem with the dataset is that it is far too small. Image data is exceedingly complex, and anything an AI could generate would push imagination to its limit. While this dataset was good, it is far from complete. We would need millions of similar images in order to train an AI on the complexities of human emotion evoked by images. Additionally, sentiment from image data, like art, is subjective rather than objective. Take a random image from that dataset for an example. It was a picture of a run down building covered in grafitti, and was labeled "negative". However, this building can carry a different sentiment based on the viewer. While some may perceive it as being dangerous, and worthy of the "negative" score, others may appreciate the stark beauty of the grafitti on an urban canvas, and they would rate it as "positive". An AI, which carries no feelings or emotions, would have to be taught to recognize numerous items, artifacts, and objects in the image, and understand the context behind why people may score it differently. Training AI to recognize and understand context, as well as make reasonable assumptions to understand and emulate human emotion is a very important subject for areas like marketing research, political science, sociology and humanities studies. Further research needs to be taken on this subject, and this (as we soon realized) is far beyond the scope of our initial project.

## Update Log
April 10: Created Github Repository, complete with gitignore and README, shared it with everyone.

April 11: Added Omar's initial data cleaning, collection, and analysis to the repo.

April 14: README update, work done towards the emoji bot.
