import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import os


# Load the model and vectorizer from the specified folder path
model_filename = 'mental_health_sentiment_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'
folder_path = 'models'

with open(os.path.join(folder_path, model_filename), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(folder_path, vectorizer_filename), 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    """
    Preprocesses the text by converting to lowercase, removing URLs, mentions, hashtags, and punctuations.

    Args:
        text: The text to be preprocessed.

    Returns:
        The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit app
def main():
    st.title("Mental Health Sentiment Analysis")

    # Input statement
    statement = st.text_area("Enter a statement:", "")

    if st.button("Analyze"):
        if statement:
            # Preprocess the statement
            cleaned_statement = preprocess_text(statement)

            # Vectorize the statement
            X = vectorizer.transform([cleaned_statement])

            # Make prediction
            prediction = model.predict(X)[0]

            # Display results
            st.write("**Predicted Mental Health Status:**", prediction)

            # Optionally, you can add more context or information about the predicted status here.
        else:
            st.warning("Please enter a statement to analyze.")

if __name__ == '__main__':
    main()


"""

Explanation

Import Libraries: Imports necessary libraries: streamlit, pandas, and pickle for loading the model.
Load Model and Vectorizer: Loads the pre-trained model and TF-IDF vectorizer using pickle.
Preprocessing Function: Defines a function to preprocess the input text.
Streamlit App:
Sets the title of the app
Creates a text area for user input (statement)
Adds an "Analyze" button
When the button is clicked:
Checks if statement is not empty
Preprocesses the statement
Vectorizes the statement using the loaded vectorizer
Makes a prediction using the loaded model
Displays the predicted mental health status
If statement is empty, shows a warning message


"""
