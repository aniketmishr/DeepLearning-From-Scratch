import streamlit as st
from config import get_config
from inference import infer_model

def translate(src_text):
    config = get_config()
    tgt_text = infer_model(config, src_text)
    return tgt_text

st.set_page_config(page_title="English to Hindi Translator", page_icon="🌍", layout="centered")

st.title("🌍 English to Hindi Translator")
st.markdown("Enter English text below and get the Hindi translation.")

src_text = st.text_area("Enter English Text:", "", height=150)

if st.button("Translate", use_container_width=True):
    if src_text.strip():
        tgt_text = translate(src_text)
        st.text_area("Hindi Translation:", tgt_text, height=150, key="translated_text")
    else:
        st.warning("Please enter some text to translate.")
