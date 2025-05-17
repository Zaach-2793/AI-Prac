import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import word_tokenize
import argparse

# ===== Define the same classes as in train.py =====

# --- For tokenization ---
def tokenize_with_math(text):
    # Add spaces around $...$ blocks
    text = re.sub(r"(\$[^$]+\$)", r" \1 ", text)
    # Replace newline/control chars
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return word_tokenize(text.lower())

# --- Vocabulary class ---
class Vocab:
    def __init__(self, max_size=20000):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]
        self.max_size = max_size

    def build(self, texts):
        counter = Counter()
        for text in texts:
            tokens = tokenize_with_math(text)
            counter.update(tokens)
        for word, _ in counter.most_common(self.max_size - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, text):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in tokenize_with_math(text)]

    def __len__(self):
        return len(self.idx2word)

# --- RNN Model ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=2
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(rnn_out_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(rnn_out_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual_proj = nn.Linear(rnn_out_dim * 2, hidden_dim)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, _ = self.rnn(embedded)
        mean_pool = torch.mean(output, dim=1)
        max_pool, _ = torch.max(output, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.ln(pooled)

        residual = self.residual_proj(pooled)
        x = self.relu(self.fc1(self.dropout(pooled)))
        x = x + residual
        out1 = self.fc2(self.dropout(x))
        out2 = self.fc2(self.dropout(x))
        return (out1 + out2) / 2

# ===== Prediction Script =====

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model components
print("🔄 Loading vocab, label encoder, and model...")
try:
    # For missing Counter in Vocab deserialization
    from collections import Counter
    
    vocab = joblib.load("vocab_rnn_model_new.pkl")
    label_encoder = joblib.load("encoder_labels_rnn_new.pkl")
    num_classes = len(label_encoder.classes_)

    model = RNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=128,
        output_dim=num_classes
    )
    model.load_state_dict(torch.load("rnn_new_model.pt", map_location=device))
    model.eval()
    model.to(device)

    print("✅ Model loaded. Ready for prediction.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise e

# Main interaction loop
while True:
    print("\n📄 Enter research paper information")
    title = input("Title: ").strip()
    summary = input("Abstract: ").strip()

    if not title and not summary:
        print("🚫 Title and abstract cannot both be empty.")
        continue

    text = f"{title} {summary}"
    tokens = vocab.encode(text)
    tokens = tokens[:256] + [0] * max(0, 256 - len(tokens))
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        probs = F.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, k=min(3, outputs.shape[1]))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        top_labels = label_encoder.inverse_transform(top_indices)

    print("\n🔍 Top predictions:")
    for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
        print(f"  {i+1}. {label} (confidence: {prob*100:.1f}%)")

    cont = input("\nWould you like to classify another paper? (y/n): ").strip().lower()
    if cont != "y":
        print("👋 Goodbye.")
        break