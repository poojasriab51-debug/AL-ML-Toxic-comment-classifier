import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = tf.keras.models.load_model("models/toxic_comment_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Toxic Comment Detection", page_icon="🛡️", layout="centered")

st.title("🛡️ Toxic Comment Detection")
st.markdown("Detect toxic content in comments across 6 categories using a deep learning model.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a **Bidirectional LSTM** model trained on the Toxic Comment Classification dataset. "
    "It classifies comments into 6 toxicity categories: **Toxic**, **Severe Toxic**, **Obscene**, "
    "**Threat**, **Insult**, and **Identity Hate**."
)

# Input
comment = st.text_area("Enter a comment to analyze:", height=150, placeholder="Type or paste a comment here...")

# Labels
labels = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]

if st.button("🔍 Analyze Comment", type="primary"):
    if comment.strip():
        with st.spinner("Analyzing comment..."):
            # Preprocess and predict
            cleaned = clean_text(comment)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=200, padding="post", truncating="post")
            predictions = model.predict(padded, verbose=0)[0]

        st.subheader("Results")
        col1, col2 = st.columns(2)

        for i, (label, prob) in enumerate(zip(labels, predictions)):
            col = col1 if i % 2 == 0 else col2
            score = round(float(prob) * 100, 1)

            if prob > 0.5:
                col.error(f"**{label}**: {score}% ⚠️")
            elif prob > 0.3:
                col.warning(f"**{label}**: {score}%")
            else:
                col.success(f"**{label}**: {score}% ✅")

        # Overall toxicity gauge
        max_prob = max(predictions)
        st.subheader("Overall Toxicity Level")
        if max_prob > 0.5:
            st.error(f"⚠️ **High Toxicity Detected** ({(max_prob * 100):.1f}%)")
        elif max_prob > 0.3:
            st.warning(f"⚠️ **Moderate Toxicity Detected** ({(max_prob * 100):.1f}%)")
        else:
            st.success(f"✅ **Comment appears clean** ({(max_prob * 100):.1f}%)")
    else:
        st.warning("Please enter a comment to analyze.")

st.markdown("---")
st.caption("Built with TensorFlow, Streamlit & ❤️")
