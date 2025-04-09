import streamlit as st
import joblib
import pickle
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt
import time

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

model = joblib.load("Logistic_Regression.pkl")

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"let's", "let us", phrase)
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    return phrase

def preprocess_tweet(text):
    text = decontracted(text)
    text = re.sub(r"\S*\d\S*", "", text).strip()
    text = re.sub(r"[^A-Za-z]+", " ", text)
    text = text.replace("#", "").replace("_", " ")
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return " ".join(tokens)

def extract_features(clean_text, raw):
    tfidf_input = vectorizer.transform([clean_text])
    sentiment = TextBlob(clean_text).sentiment.polarity
    tweet_len = len(clean_text)
    num_hashtags = raw.count("#")
    has_mention = int("@" in raw)
    extra_feat = np.array([[sentiment, tweet_len, num_hashtags, has_mention]])
    return hstack([tfidf_input, extra_feat])

# UI Setup
st.set_page_config(page_title="Disaster Tweet Detector", layout="centered")

if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""

def set_example(text):
    st.session_state["tweet_input"] = text

st.markdown("""
    <div style='text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 10px;'>
        <h1>🌪️ Disaster Tweet Detector</h1>
        <p style='font-size:16px;'>Classify tweets as <b style='color:red;'>Disaster</b> 🚨 or <b style='color:green;'>Non-Disaster</b> ✅</p>
    </div>
""", unsafe_allow_html=True)

# Try an example section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'><b>🔁 Try an example</b></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px;'>Click any button below to auto-fill the input box.</p>", unsafe_allow_html=True)

ex1, ex2 = st.columns([1, 1])
with ex1:
    if st.button("✅ Puppy Tweet 🐶", use_container_width=True):
        set_example("Just got a new puppy, he's so cute! 🐾")
with ex2:
    if st.button("🔥 Fire Alert 🚨", use_container_width=True):
        set_example("Forest fire spreading rapidly in California! 🔥")

# Input box
st.markdown("<div style='text-align:center;'><label style='font-size:16px;font-weight:bold;'>✍️ Enter a tweet to classify:</label></div>", unsafe_allow_html=True)
tweet = st.text_area("", value=st.session_state["tweet_input"], height=100, label_visibility="collapsed", key="tweet_input")

# Center aligned buttons
btn1, btn2, btn3 = st.columns([2, 3, 2])
with btn2:
    predict_clicked = st.button("🔍 Predict Tweet", use_container_width=True)
    reset_clicked = st.button("🧹 Clear Input", use_container_width=True)

if reset_clicked:
    st.session_state["tweet_input"] = ""
    st.experimental_rerun()

# Prediction logic
if predict_clicked:
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        clean_text = preprocess_tweet(tweet)
        features = extract_features(clean_text, tweet)
        prediction_proba = model.predict_proba(features)[0]
        prediction = int(np.argmax(prediction_proba))
        confidence = prediction_proba[prediction]
        sentiment_score = TextBlob(clean_text).sentiment.polarity

        if sentiment_score < -0.1:
            mood = "🔥 Urgent Tone"
        elif sentiment_score > 0.2:
            mood = "😊 Positive Vibes"
        else:
            mood = "😶 Mild/Neutral Tone"

        with st.spinner("Analyzing tweet..."):
            time.sleep(1.5)

        # Display result
        st.markdown(f"""
        <div style='text-align:center; border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px auto; max-width: 600px;'>
            <h2 style='color:#0099ff;'>📢 Prediction Result</h2>
            <div style='font-size:20px; color:{"red" if prediction == 1 else "green"};'>
                {"🚨 <b>Disaster</b>" if prediction == 1 else "✅ <b>Non-Disaster</b>"} <span style='font-size:16px;'>(Confidence: {confidence:.2%})</span>
            </div>
            <div style='margin-top: 5px;'>{'🛑 Likely emergency or natural disaster content.' if prediction == 1 else '☑️ Likely personal or casual tweet.'}</div>
            <div style='margin-top: 10px; font-style: italic; color: gray;'>{mood}</div>
        </div>
        """, unsafe_allow_html=True)

        # Additional Insights
        st.markdown("""
        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px auto; max-width: 900px;'>
            <h3 style='text-align:center;'>🧠 Additional Insights</h3>
            <p style='text-align:center;font-size:14px;'>Below is a breakdown of model confidence and tweet-specific characteristics.</p>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
                <div style='padding:10px; border: 1px solid #eee; border-radius: 10px;'>
                    <h4 style='text-align:center;'>📈 Confidence Breakdown</h4>
                    <p style='font-size:13px;text-align:center;'>This pie chart shows how confident the model is about each class.</p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.pie(prediction_proba, labels=["Non-Disaster", "Disaster"], autopct="%1.1f%%", colors=["#8BC34A", "#FF5252"])
            ax.axis("equal")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            emoji_pattern = re.compile("[\U0001F1E0-\U0001F9FF\U0001F300-\U0001F6FF\u2600-\u26FF]")
            emoji_count = len(emoji_pattern.findall(tweet))
            word_count = len(tweet.split())
            avg_word_len = round(sum(len(w) for w in tweet.split()) / word_count, 2) if word_count else 0
            st.markdown(f"""
                <div style='padding:10px; border: 1px solid #eee; border-radius: 10px;'>
                    <h4 style='text-align:center;'>📊 Tweet Analysis</h4>
                    <p style='font-size:13px;text-align:center;'>Here's a breakdown of tweet-specific features used to help with prediction.</p>
                    <ul>
                        <li><b>🧠 Sentiment Score:</b> {sentiment_score:.3f}</li>
                        <li><b>📝 Tweet Length:</b> {len(clean_text)} characters</li>
                        <li><b>#️⃣ Hashtags Count:</b> {tweet.count("#")}</li>
                        <li><b>👥 Mentions Present:</b> {int("@" in tweet)}</li>
                        <li><b>😊 Emoji Count:</b> {emoji_count}</li>
                        <li><b>🔤 Avg Word Length:</b> {avg_word_len}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("🤖 Powered by Logistic Regression | Features: TF-IDF + Sentiment + Length + Hashtags + Mentions + Emoji Count + Avg Word Length")
