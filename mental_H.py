import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re

# Load the model and vectorizer
model = pickle.load(open('models/mental_health_sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Streamlit app
def main():
    # Display the image (replace 'mental_health_app_image.jpg' with your actual image path)
    st.image("image0.jpg", caption="Mental Health Patients App by Jayymadd Clicke", use_column_width=True)

    st.title("Mental Health Sentiment Analysis")

    # Input statement
    statement = st.text_area("Enter a statement:", "")

    if st.button("Analyze"):
        if statement:
            # Preprocess and predict
            cleaned_statement = preprocess_text(statement)
            X = vectorizer.transform([cleaned_statement])
            prediction = model.predict(X)[0]

            # Display results
            st.write("**Predicted Mental Health Status:**", prediction)

        else:
            st.warning("Please enter a statement to analyze.")

    # Add contact information
    st.markdown("---")
    st.subheader("Contact for Support:")
    st.write("Email: jayymaddclicke@gmail.com")
    st.write("Phone: 08024621105")

    # Add donation link (replace with your actual link)
    st.markdown("---")
    st.subheader("Support Mental Health Initiatives:")
    st.markdown("[Donate Now](https://www.yourdonationlink.com)")  

if __name__ == '__main__':
    main()


"""
sentiment analysis of mental health to predict Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder 

"""
