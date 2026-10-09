"""
Try the saved model WITHOUT training again.
Needs my_model.keras and tokenizer.json (created by sentiment_analysis.py).
"""

import os
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 300  # must match sentiment_analysis.py

if not (os.path.exists("my_model.keras") and os.path.exists("tokenizer.json")):
    sys.exit("my_model.keras / tokenizer.json not found.\n"
             "Run 'python sentiment_analysis.py' first to train and save them.")

model = load_model("my_model.keras")

with open("tokenizer.json", "r", encoding="utf-8") as file:
    tokenizer = tokenizer_from_json(file.read())

print("========== MOVIE REVIEW SENTIMENT ANALYZER ==========")
print("Enter a movie review to analyze. Type 'exit' to quit.")

while True:
    text = input("\nEnter movie review: ").strip()

    if text.lower() == "exit":
        break
    if not text:
        print("Please enter a review.")
        continue

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])

    print(f"Positive score: {score:.4f}")
    print("Sentiment:", "POSITIVE" if score >= 0.5 else "NEGATIVE")
