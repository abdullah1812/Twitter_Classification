import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np

# Load model and tokenizer
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(r'Projects\Classification\Twitter\twitter_lstm_sentiment_model.h5')
    with open(r'Projects\Classification\Twitter\tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def clean_txt(text):
  text = text.lower()
  text = re.sub(r'http\S+|www\S+|https\S+', '', text)
  text = re.sub(r'\@\w+|\#', '', text)
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  text = re.sub(r'\d+', '', text)
  return text

# Prediction function
def predict_sentiment(tweet, max_len=100):
    cleaned = clean_txt(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)[0][0]
    p = np.argmax(pred)
    if p == 0 :
      sentiment = 'Irrelevant'
    elif p ==1:
      sentiment = 'Negative'
    elif p ==2:
      sentiment = 'Neutral'
    elif p ==3:
      sentiment = 'Positive'

    return sentiment, pred

# Set page configuration for a wider layout and custom theme
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #1a1a2e;
        color: #e0e0e0;
        padding: 20px;
    }
    .title {
        color: #00d4ff;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        color: #a9a9b2;
        text-align: center;
        font-size: 18px;
        font-style: italic;
    }
    .stTextInput > div > div > input {
        background-color: #2e2e4a;
        color: #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #444;
    }
    .stButton > button {
        background-color: #00d4ff;
        color: #1a1a2e;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00b4d8;
        color: #1a1a2e;
    }
    .result {
        color: #00ff95;
        text-align: center;
        font-size: 20px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown('<div class="subtitle">Enter a tweet to predict its sentiment (Positive, Negative, Neutral, or Irrelevant).</div>', unsafe_allow_html=True)

# Add some vertical spacing
st.markdown("<br>", unsafe_allow_html=True)

# Styled input label and input box
user_input = st.text_input("Type your tweet here...", placeholder="E.g., I love this sunny day! 😊")
st.markdown("<br>", unsafe_allow_html=True)

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        sentiment, score = predict_sentiment(user_input)
        print(sentiment)
        st.success(f"**Sentiment**: {sentiment}")

    else:
        st.error("Please enter a tweet.")