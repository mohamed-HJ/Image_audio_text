import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample text data
text_data = [
    "Text summarization is the process of distilling the most important information from a source text.",
    "There are two main approaches to text summarization: extractive and abstractive.",
    "Extractive summarization involves selecting important sentences or phrases from the original text.",
    "Abstractive summarization involves generating new sentences to capture the essence of the original text."
]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(text_data)

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Calculate importance scores (e.g., based on word frequency or other metrics)
importance_scores = np.mean(padded_sequences, axis=1)

# Select top sentences based on importance scores
num_sentences_to_keep = 2
top_indices = importance_scores.argsort()[-num_sentences_to_keep:][::-1]

# Generate summary
summary = [text_data[i] for i in top_indices]
print("Summary:")
for sentence in summary:
    print("-", sentence)
