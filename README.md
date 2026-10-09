# 🎬 Movie Review Sentiment Analysis

A deep learning project that reads a movie review and predicts whether it is **positive** or **negative**. It uses a **1D Convolutional Neural Network (CNN)** built with TensorFlow / Keras and trained on the IMDb movie review dataset.

---

## 📌 Overview

Sentiment analysis is a common Natural Language Processing (NLP) task: deciding the opinion expressed in a piece of text. In this project:

1. Reviews are converted into sequences of numbers (tokenization and padding).
2. A CNN learns which word patterns point to a positive or a negative opinion.
3. The trained model scores any review you type, from 0 (negative) to 1 (positive).

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| Name | IMDb Large Movie Review Dataset |
| Source | Hugging Face `datasets` library (`load_dataset("imdb")`) |
| Training reviews | 25,000 (split into 90% training and 10% validation) |
| Test reviews | 25,000 |
| Labels | 0 = negative, 1 = positive |

The dataset is downloaded automatically the first time you run the project, so an internet connection is needed once.

---

## 🧠 Model Architecture

| Layer | Details | Purpose |
|-------|---------|---------|
| Input | 300 tokens per review | Fixed-length review |
| Embedding | 20,000 words, 128 dimensions | Turns word numbers into meaningful vectors |
| Conv1D | 128 filters, kernel size 5, ReLU | Detects important 5-word patterns |
| GlobalMaxPooling1D | | Keeps the strongest signal from each filter |
| Dense | 32 units, ReLU | Combines the signals |
| Dropout | 0.5 | Reduces overfitting |
| Dense | 1 unit, sigmoid | Outputs the positive score (0 to 1) |

**Training settings:** Adam optimizer, binary cross-entropy loss, batch size 32, up to 8 epochs, with Early Stopping (monitors validation loss, patience 2, restores the best weights).

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| TensorFlow / Keras | Building and training the CNN |
| Hugging Face `datasets` | Loading the IMDb dataset |
| NumPy | Array handling |

---

## 📂 Project Structure

```
movie-review-sentiment-analysis/
├── sentiment_analysis.py   # Train, evaluate, save, and try the model
├── predict.py              # Use the saved model without training again
├── requirements.txt
├── .gitignore
└── README.md
```

After training, two more files are created in the same folder:

| File | Contents |
|------|----------|
| `my_model.keras` | The trained model |
| `tokenizer.json` | The word-to-number mapping used by the model |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/fatima99tariq-cpu/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS / Linux use `source venv/bin/activate`.

### 3. Install the libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Option 1: Train the model and try it

```bash
python sentiment_analysis.py
```

The script will:

1. Download the dataset
2. Train the CNN
3. Print training, validation and test accuracy and loss
4. Save `my_model.keras` and `tokenizer.json`
5. Ask you to type reviews

Training takes a few minutes on a normal laptop CPU.

### Option 2: Use the saved model (no training)

After Option 1 has created the two saved files:

```bash
python predict.py
```

### Example

```
Enter movie review: This movie was amazing, I loved every minute of it
Positive score: 0.9xxx
Sentiment: POSITIVE

Enter movie review: Waste of time, the story was boring and the acting was terrible
Positive score: 0.0xxx
Sentiment: NEGATIVE
```

Type `exit` to quit.

---

## 🔬 How It Works

1. **Load** the IMDb dataset and shuffle the training data.
2. **Tokenize** every review, keeping the 20,000 most common words (unknown words become `<OOV>`).
3. **Pad or cut** every review to exactly 300 tokens.
4. **Train** the CNN with early stopping on validation loss.
5. **Evaluate** the best model on the training, validation and test data.
6. **Save** the model and tokenizer, so the same word mapping can be reused later.
7. **Predict:** a new review goes through the same tokenizer and padding, and the model returns a score. A score of 0.5 or more means positive.

---

## ⚠️ Limitations

- Works for **English** reviews only.
- Only the first 300 words of a review are used.
- Sarcasm and mixed opinions can fool the model.
- It gives only positive or negative, with no neutral class.
- Words that are not among the 20,000 most common are treated as unknown.

---

## 🔮 Future Improvements

- Try an LSTM, GRU or Bidirectional LSTM and compare with the CNN
- Use pre-trained embeddings (GloVe) or a Transformer such as BERT
- Add a neutral class
- Build a simple web interface with Streamlit or Flask
- Plot the accuracy and loss curves

---

## 👩‍💻 Author

GitHub: [fatima99tariq-cpu](https://github.com/fatima99tariq-cpu)
