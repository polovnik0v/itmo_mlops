import streamlit as st
import requests
import time

ENDPOINT = "http://localhost:8000/predict"

st.title("Sentiment Classifier")
st.caption("Модель загружена из ClearML Registry")

text = st.text_area("Введи текст на английском", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Введи текст")
    else:
        try:
            t0 = time.time()
            r = requests.post(ENDPOINT, json={"text": text}, timeout=5)
            ms = (time.time() - t0) * 1000

            result = r.json()
            label = result["label"]
            confidence = result["confidence"]

            if label == "positive":
                st.success(f"Метка: **{label}**  —  уверенность: {confidence:.0%}")
            else:
                st.error(f"Метка: **{label}**  —  уверенность: {confidence:.0%}")

            st.caption(f"Latency: {ms:.1f} ms")

        except requests.exceptions.ConnectionError:
            st.error("Endpoint недоступен. Запусти uvicorn serve:app --port 8000")
        except Exception as e:
            st.error(f"Ошибка: {e}")