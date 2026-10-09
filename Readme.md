<div align="center">

# 🎬 Movie Review Sentiment Analysis
### Deep Learning Practical Report 5 — RNN, LSTM & GRU

**A deep learning project to classify IMDB movie reviews as Positive or Negative.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Task](https://img.shields.io/badge/Task-Binary%20Text%20Classification-purple)
![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K-green)

</div>

---

## 📌 Project Overview

This project uses recurrent neural networks to predict the sentiment of movie reviews. It compares a basic **SimpleRNN**, **LSTM**, and **GRU** model and explores how sequence order, memory, and gradient flow affect sentiment classification.

- **Institute:** Red & White Skill Education
- **Subject:** Deep Learning
- **Practical:** PR 5
- **Problem type:** Binary text classification
- **Output classes:** Positive and Negative

## 🧹 Data Preprocessing

The notebook prepares raw text for the models using these steps:

1. Load and inspect the CSV file.
2. Check missing values and remove duplicate rows.
3. Convert text to lowercase and remove HTML tags and unwanted characters.
4. Keep apostrophes and important words such as **“not”** and **“no”**, because they can change a review's meaning.
5. Split data into training and testing sets using an 80/20 split.
6. Tokenise text using a Keras `Tokenizer` with a 10,000-word vocabulary and an out-of-vocabulary token.
7. Convert words into integer sequences and pad/truncate them to a consistent length.

## 🧠 Models Used

### 1. SimpleRNN
SimpleRNN reads a review one time step at a time and carries a hidden state through the sequence. It provides a baseline for understanding recurrent networks, but it can struggle to retain information over long sequences because of vanishing gradients.

### 2. LSTM
Long Short-Term Memory (LSTM) uses forget, input, and output gates plus a cell state to manage information over time. It is designed to preserve useful information over longer sequences and reduce the vanishing-gradient problem.

### 3. GRU
Gated Recurrent Unit (GRU) uses update and reset gates without a separate cell state. It is a simpler gated architecture that can provide strong performance with fewer parameters than a comparable LSTM.

### Additional experiments
- ANN text baseline using an Embedding and Flatten layer
- Manual RNN forward propagation using NumPy and verification with Keras
- Backpropagation Through Time (BPTT)
- Vanishing and exploding gradient experiments
- Gradient clipping
- Review-length robustness
- LSTM/GRU hidden-unit sensitivity
- Misclassification analysis and custom sentence predictions

## 📊 Model Results

The table below records selected evaluation results shown in the notebook. Results can vary with hardware, random seeds, and training conditions.

| Model | Test Accuracy | ROC-AUC |
|---|---:|---:|
| SimpleRNN | 79.23% | 0.8627 |
| LSTM | 88.45% | 0.9480 |
| GRU | 88.50% | 0.9488 |

**Observation:** In these recorded runs, GRU achieved the highest test accuracy among the three listed recurrent models, narrowly ahead of LSTM. LSTM and GRU both performed better than SimpleRNN. Refer to the notebook's full results table for additional metrics and experiments.

## 📈 Visualisations

The notebook includes plots and tables for:

- Review-length distribution and sentiment-wise review lengths
- Manual RNN hidden-state heatmap
- Training and validation loss/accuracy
- Gradient norms across time steps
- Sequence-length accuracy comparison
- Confusion matrices and ROC curves
- Model comparison and hidden-unit sensitivity
- Accuracy by short, medium, and long reviews

If you export plots from the notebook, place them in a `plots/` folder and add them to this README as needed. Example:

## 🔎 Prediction / Inference

The notebook includes a `predict_sentiment(text)` function. It applies the same text-cleaning, tokenisation, and padding steps used during training, then returns a predicted sentiment and probability.

Example custom inputs include positive, negative, short, negated, and sarcastic sentences. Predictions are model estimates and may be incorrect, especially for sarcasm, mixed opinions, and complex negation.

## 🏁 Conclusion

This project demonstrates that recurrent architectures can use word order and context to classify movie-review sentiment. In the recorded notebook runs, LSTM and GRU achieved stronger results than SimpleRNN. GRU is a practical candidate when balancing performance and model complexity, while the final choice should be based on the complete evaluation results and inference-time requirements.

### Limitations
- The word embeddings are learned from this dataset rather than from pretrained language models.
- Sarcasm, subtle context, and complicated negation can still be difficult to classify.
- Results depend on the chosen vocabulary size, padding length, and training setup.

## 🛠️ Tools & Libraries

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn

## 📁 Project Structure

```text
Movie-Review-Sentiment-Analysis/
├── DL_PR5.ipynb
├── DL_PR5.html
├── IMDB Dataset.csv          # Download separately from Kaggle
├── requirements.txt
├── plots/
│   ├── length_distribution.png
│   ├── rnn_forward_heatmap.png
│   ├── gradient_norms_all_cells.png
│   ├── training_curves_rnn_lstm_gru.png
│   ├── roc_curves.png
│   ├── results_table.png
│   └── accuracy_by_length.png
└── README.md
```

## 🎥 Project Video

**Video explanation:** `PASTE_YOUR_GOOGLE_DRIVE_OR_UNLISTED_YOUTUBE_LINK_HERE`

Upload your 5–10 minute project recording to Google Drive (set access to “Anyone with the link”) or YouTube (Unlisted), then replace the placeholder above with the working URL.

## 👩‍💻 Author

**Janki Dholariya** 
