
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(r'/content/drive/MyDrive/NLP/Projects/Twitter Sentiment Analysis/twitter_lstm_sentiment_model.h5')
    with open(r'/content/drive/MyDrive/NLP/Projects/Twitter Sentiment Analysis/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Text cleaning function (matching notebook preprocessing)
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()

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
#     Categories and their labels:
# Label 0: Irrelevant
# Label 1: Negative
# Label 2: Neutral
# Label 3: Positive

# Streamlit app layout
st.title("Twitter Sentiment Analysis with LSTM")
st.write("Enter a tweet to predict its sentiment (positive, negative, Netural or Irrelevant).")

# User input
user_input = st.text_area("Tweet", "Type your tweet here...", height=100)

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        sentiment, score = predict_sentiment(user_input)
        st.success(f"**Sentiment**: {sentiment}")
        # st.write(f"**Confidence Score**: {score:.2%}")

        # Bar chart for sentiment scores
        fig, ax = plt.subplots()
        scores = [1 - score, score]
        labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
        sns.barplot(x=scores, y=labels, palette=['#FF6B6B', '#4ECDC4'])
        ax.set_xlabel('Probability')
        ax.set_title('Sentiment Prediction Confidence')
        st.pyplot(fig)
    else:
        st.error("Please enter a tweet.")
