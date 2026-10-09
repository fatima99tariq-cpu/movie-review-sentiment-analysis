"""
Movie Review Sentiment Analysis (IMDb) using a 1D CNN.

Run this file to:
  1. download the IMDb dataset
  2. train the model
  3. print train / validation / test results
  4. save the model (my_model.keras) and tokenizer (tokenizer.json)
  5. let you type your own reviews
"""

import numpy as np
from datasets import load_dataset

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping

VOCAB_SIZE = 20000
MAX_LENGTH = 300
SEED = 42

# ----------------------------
# 1. LOAD DATASET
# ----------------------------

print("\nLoading IMDb movie review dataset...")

dataset = load_dataset("imdb")

train_texts = dataset["train"]["text"]
train_labels = np.array(dataset["train"]["label"], dtype="float32")

test_texts = dataset["test"]["text"]
test_labels = np.array(dataset["test"]["label"], dtype="float32")

print("Dataset loaded successfully!")
print("Training reviews:", len(train_texts))
print("Testing reviews:", len(test_texts))

# The IMDb training split is sorted: all negative reviews first, then all
# positive ones. Shuffle it, otherwise the validation set would contain
# positive reviews only.
rng = np.random.default_rng(SEED)
order = rng.permutation(len(train_texts))
train_texts = [train_texts[i] for i in order]
train_labels = train_labels[order]

# ----------------------------
# 2. TEXT PREPROCESSING
# ----------------------------

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_all = pad_sequences(
    tokenizer.texts_to_sequences(train_texts),
    maxlen=MAX_LENGTH, padding="post", truncating="post",
)
X_test = pad_sequences(
    tokenizer.texts_to_sequences(test_texts),
    maxlen=MAX_LENGTH, padding="post", truncating="post",
)

# 90% train / 10% validation (data is already shuffled)
val_size = int(0.1 * len(X_all))
X_val, y_val = X_all[:val_size], train_labels[:val_size]
X_train, y_train = X_all[val_size:], train_labels[val_size:]

print("Text preprocessing completed!")
print("Train:", len(X_train), "| Validation:", len(X_val), "| Test:", len(X_test))

# ----------------------------
# 3. BUILD CNN MODEL
# ----------------------------

model = Sequential([
    Input(shape=(MAX_LENGTH,)),
    Embedding(input_dim=VOCAB_SIZE, output_dim=128),
    Conv1D(128, 5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nCNN model built successfully!")
model.summary()

# ----------------------------
# 4. TRAIN MODEL
# ----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

print("\nTraining started...")

history = model.fit(
    X_train,
    y_train,
    epochs=8,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1,
)

print("\nTraining completed!")

# ----------------------------
# 5. EVALUATE MODEL
# ----------------------------
# EarlyStopping restores the BEST weights, so all three numbers below are
# measured on the same final model (not on the last epoch).

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, test_labels, verbose=0)

print("\n========== FINAL RESULTS ==========")
print(f"Training Accuracy:   {train_accuracy:.4f}   (loss {train_loss:.4f})")
print(f"Validation Accuracy: {val_accuracy:.4f}   (loss {val_loss:.4f})")
print(f"Test Accuracy:       {test_accuracy:.4f}   (loss {test_loss:.4f})")

# ----------------------------
# 6. SAVE MODEL AND TOKENIZER
# ----------------------------
# Saved BEFORE the interactive part, so nothing is lost if you close the
# program while typing reviews.

model.save("my_model.keras")

with open("tokenizer.json", "w", encoding="utf-8") as file:
    file.write(tokenizer.to_json())

print("\nModel saved as my_model.keras")
print("Tokenizer saved as tokenizer.json")

# ----------------------------
# 7. PREDICT MOVIE REVIEW
# ----------------------------

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence, maxlen=MAX_LENGTH, padding="post", truncating="post"
    )
    prediction = float(model.predict(padded, verbose=0)[0][0])

    print(f"Positive score: {prediction:.4f}")
    if prediction >= 0.5:
        print("Sentiment: POSITIVE")
    else:
        print("Sentiment: NEGATIVE")


# ----------------------------
# 8. USER INPUT
# ----------------------------

print("\n========== MOVIE REVIEW SENTIMENT ANALYZER ==========")
print("Enter a movie review to analyze.")
print("Type 'exit' to quit.")

while True:
    text = input("\nEnter movie review: ").strip()

    if text.lower() == "exit":
        break

    if not text:
        print("Please enter a review.")
        continue

    predict_sentiment(text)

print("\nProject finished successfully!")
