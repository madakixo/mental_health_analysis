## Streamlit App  link[https://mentalstatecheck.streamlit.app/]

# Mental Health Sentiment Analysis App

This Streamlit application provides a simple tool for analyzing the sentiment of text input and predicting potential mental health statuses like Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, and Personality Disorder. 

## How it works

*   The app utilizes a pre-trained machine learning model that has been fine-tuned on a dataset of labeled statements related to mental health.
*   The model uses TF-IDF vectorization to convert text input into numerical features and then makes a prediction based on these features.
*   The predicted mental health status is displayed to the user.

## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

4.  **Enter a statement in the text area and click "Analyze".**
5.  **The app will display the predicted mental health status.**

## Important Notes

*   This app is intended for informational and educational purposes only. It is not a substitute for professional medical advice or diagnosis.
*   The model's predictions are based on patterns learned from the training data and may not always be accurate.
*   If you or someone you know is struggling with mental health issues, please seek help from a qualified mental health professional.
*   **Contact Information:**
    *   Email: jayymaddclicke@gmail.com
    *   Phone: 08024621105

*   **Support Mental Health Initiatives:**
    *   [Donate Now](https://www.yourdonationlink.com) (Replace with your actual donation link)

## Model Details

*   The model is a Logistic Regression classifier trained on a combined dataset of mental health statements.
*   The model and TF-IDF vectorizer are saved in the `models` folder as `mental_health_sentiment_model.pkl` and `tfidf_vectorizer.pkl`, respectively.
*   You can retrain the model with new data or experiment with different algorithms if needed.

## Contributing

Contributions to improve the app or the underlying model are welcome! Please feel free to open issues or pull requests on the repository.

## License

This project is licensed under the [MIT License](LICENSE).

**Disclaimer:** 

This app is not intended to be a diagnostic tool or a replacement for professional medical advice. If you are concerned about your mental health, please consult a qualified healthcare provider. 

