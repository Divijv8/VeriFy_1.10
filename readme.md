# 📰 Fake News Detection using Deep Learning + Gemini AI

## 📘 Overview
This project detects fake news articles using a trained deep learning model
and provides factual explanations powered by Google Gemini.

## 🧩 Tech Stack
- Python, TensorFlow, Pandas, Scikit-learn
- Google Generative AI (Gemini 2.5 Flash)
- Streamlit (for UI)

## ⚙️ How It Works
1. Preprocess datasets (Fake1.csv, True1.csv, etc.)
2. Train an LSTM model for fake news classification
3. Use Gemini API to explain predictions in plain English

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python app/main.py
