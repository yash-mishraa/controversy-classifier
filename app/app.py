import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Load model and embedder
model = joblib.load("models/model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
analyzer = SentimentIntensityAnalyzer()

# Label map
label_map = {
    0: ("🟢 Safe", "This text is considered non-controversial."),
    1: ("🟡 Mildly Divisive", "This text shows signs of being somewhat polarizing."),
    2: ("🔴 Highly Controversial", "This text is likely to trigger strong disagreement.")
}

# Title
st.set_page_config(page_title="Controversy Classifier", page_icon="🔥")
st.title("🔥 Controversy Classifier")
st.write("Enter a tweet, news headline, or statement to check how controversial it is.")

# Text Input
text = st.text_area("✍️ Input text:", placeholder="e.g., 'This government is ruining the country.'")

if st.button("Analyze") and text.strip():
    # Extract features
    embedding = embedder.encode([text])
    sentiment = analyzer.polarity_scores(text)["compound"]
    features = np.hstack([embedding, np.array([[sentiment]])])

    # Predict
    pred = model.predict(features)[0]
    score = model.predict_proba(features)[0][pred]

    # Display
    label, explanation = label_map[pred]
    st.markdown(f"### Controversy Score: `{score:.2f}`")
    st.markdown(f"### Classification: {label}")
    st.info(explanation)

    # Explanation details
    st.markdown("---")
    st.markdown("#### 🧠 Why this result?")
    st.markdown(f"- Sentiment score: `{sentiment:.2f}`")
    st.markdown(f"- Embedding vector: `{embedding.shape[1]}-dimensional` sentence representation used")
