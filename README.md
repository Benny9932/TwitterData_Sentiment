# Sentiment Analysis on Twitter Data with RNNs

This project performs sentiment analysis on tweets using RNN-based models (GRU and LSTM) trained on the Sentiment140 dataset. The goal is to classify tweets as either positive or negative based on their text content.

## 📁 Dataset

The dataset contains 1.6 million tweets, with the following structure:

- `0` = sentiment (0 = negative, 4 = positive)
- `1` = tweet ID
- `2` = date of the tweet
- `3` = query (unused)
- `4` = user
- `5` = tweet text

The data is preprocessed and the sentiment labels are normalized to:
- `0`: Negative
- `1`: Positive

## 🧹 Preprocessing

- Removed non-alphanumeric characters
- Removed newlines
- Removed short words (1–2 characters)
- Normalized sentiment labels

## 🧠 Models

Two models were trained:

### GRU Model
- TextVectorization layer with `max_tokens=1000`
- Embedding layer
- GRU layer (128 units)
- Dense + ReLU
- Final Dense + Sigmoid

**Accuracy:** `78.6%`

### LSTM Model
- Same architecture, replacing GRU with LSTM (128 units)

**Accuracy:** `78.5%`

## 📊 Evaluation

Both models were evaluated using:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

