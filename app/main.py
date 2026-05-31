import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from utils import tokenize_training_style
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "my_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "model", "tokenizer.pkl")
MAX_LEN = 1000

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

def predict_news(text):
    tokens = tokenize_training_style(text)
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = float(model.predict(padded, verbose=0)[0][0])
    label = "Fake" if pred < 0.5 else "Real"
    return label, pred, tokens, seq, int((padded != 0).sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, help="Text to analyze (optional)")
    args = parser.parse_args()

    if args.text:
        label, pred, tokens, seq, nonzero = predict_news(args.text)
        print("Prediction:", label, f"({pred:.4f})")
        print("TOKENS:", tokens[:40])
        print("SEQ (first 40 indices):", seq[0][:40])
        print("NONZERO tokens in padded:", nonzero)
    else:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            t = input("\nEnter news (or exit): ").strip()
            if t.lower() == "exit":
                break
            label, pred, tokens, seq, nonzero = predict_news(t)
            print("Prediction:", label, f"({pred:.4f})")
            print("TOKENS:", tokens[:40])
            print("SEQ (first 40 indices):", seq[0][:40])
            print("NONZERO tokens in padded:", nonzero)
