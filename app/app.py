import streamlit as st
import requests
import os

API_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="Fake News Detection", layout="centered", page_icon=":newspaper:")
st.title("Fake News Detection (Deep Learning using LSTM)")
st.write("The trained model has an accuracy of 98.10% on the validation dataset.")

# space between title and user interface
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Paste the title of an article below to verify its authenticity using our trained RNN model. (OPTIONAL)")
title = st.text_area("Article Title", height=120, width=800, placeholder="Type or paste TITLE here...")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### *Paste the content of an article below to verify its authenticity using our trained RNN model.")
text = st.text_area("Article Content", height=300, width=800, placeholder="Type or paste CONTENT here...")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Check Authenticity"):
    if title.strip() == "" and text.strip() == "":
        st.warning("Please enter either the title or content of the article.")
    else:
        combined_input = f"{title} {text}"
        response = requests.post(f"{API_URL}", json={"text": combined_input})
        if response.status_code == 200:
            result = response.json()
            is_fake = result["is_fake"]
            confidence = result["confidence"]
            score = result["score"]

            if is_fake:
                st.error(f"⚠️ The article is likely FAKE with a confidence of {confidence:.2%}.")
            else:
                st.success(f"✅ The article is likely REAL with a confidence of {confidence:.2%}.")


