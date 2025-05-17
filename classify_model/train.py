# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from collections import Counter
# from nltk.tokenize import word_tokenize
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import f1_score, classification_report
# from constants.constants_map import cat_fullname_mapping
# import numpy as np
# import random
# import argparse
# import re
# import logging
# import string
# import joblib
# from torch.nn.functional import normalize

# # --- Clean label text ---
# def clean_label(text):
#     if pd.isna(text): return ""
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9]+", "_", text)
#     return text.strip("_")

# def tokenize_with_math(text):
#     # Add spaces around $...$ blocks
#     text = re.sub(r"(\$[^$]+\$)", r" \1 ", text)
#     # Replace newline/control chars
#     text = re.sub(r"\\[a-zA-Z]+", " ", text)
#     return word_tokenize(text.lower())

# # --- Vocab ---
# class Vocab:
#     def __init__(self, max_size=20000):
#         self.word2idx = {"<pad>": 0, "<unk>": 1}
#         self.idx2word = ["<pad>", "<unk>"]
#         self.max_size = max_size

#     def build(self, texts):
#         counter = Counter()
#         for text in texts:
#             tokens = tokenize_with_math(text)
#             counter.update(tokens)
#         for word, _ in counter.most_common(self.max_size - 2):
#             if word not in self.word2idx:
#                 self.word2idx[word] = len(self.idx2word)
#                 self.idx2word.append(word)

#     def encode(self, text):
#         return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in tokenize_with_math(text)]

#     def __len__(self):
#         return len(self.idx2word)

# # --- Supervised Dataset ---
# class TextDataset(Dataset):
#     def __init__(self, texts, labels, vocab, max_len=256):
#         self.texts = texts
#         self.labels = labels
#         self.vocab = vocab
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         tokens = self.vocab.encode(self.texts[idx])
#         tokens = tokens[:self.max_len] + [0] * max(0, self.max_len - len(tokens))
#         return {
#             'input_ids': torch.tensor(tokens, dtype=torch.long),
#             'label': torch.tensor(self.labels[idx], dtype=torch.long)
#         }

# # --- SimCLR Dataset ---
# class SimCLRTextDataset(Dataset):
#     def __init__(self, texts, vocab, max_len=256):
#         self.texts = texts
#         self.vocab = vocab
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def augment(self, text):
#         tokens = tokenize_with_math(text)
#         if len(tokens) > 4:
#             tokens = [t for t in tokens if random.random() > 0.2]
#         random.shuffle(tokens)
#         return " ".join(tokens)

#     def __getitem__(self, idx):
#         t1 = self.augment(self.texts[idx])
#         t2 = self.augment(self.texts[idx])
#         ids1 = self.vocab.encode(t1)[:self.max_len]
#         ids2 = self.vocab.encode(t2)[:self.max_len]
#         ids1 += [0] * (self.max_len - len(ids1))
#         ids2 += [0] * (self.max_len - len(ids2))
#         return {
#             "input_ids1": torch.tensor(ids1),
#             "input_ids2": torch.tensor(ids2)
#         }

# # --- RNN Model ---
# class RNNClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5, bidirectional=True):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.rnn = nn.GRU(
#             embedding_dim,
#             hidden_dim,
#             batch_first=True,
#             bidirectional=bidirectional,
#             num_layers=2
#         )
#         rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
#         self.ln = nn.LayerNorm(rnn_out_dim * 2)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc1 = nn.Linear(rnn_out_dim * 2, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.residual_proj = nn.Linear(rnn_out_dim * 2, hidden_dim)

#     def forward(self, input_ids):
#         embedded = self.dropout(self.embedding(input_ids))
#         output, _ = self.rnn(embedded)
#         mean_pool = torch.mean(output, dim=1)
#         max_pool, _ = torch.max(output, dim=1)
#         pooled = torch.cat([mean_pool, max_pool], dim=1)
#         pooled = self.ln(pooled)

#         residual = self.residual_proj(pooled)
#         x = self.relu(self.fc1(self.dropout(pooled)))
#         x = x + residual
#         out1 = self.fc2(self.dropout(x))
#         out2 = self.fc2(self.dropout(x))
#         return (out1 + out2) / 2

# # --- Pretraining helper ---
# def contrastive_loss(z1, z2, temperature=0.5):
#     z1 = normalize(z1, dim=1)
#     z2 = normalize(z2, dim=1)
#     N = z1.size(0)
#     z = torch.cat([z1, z2], dim=0)
#     sim = torch.matmul(z, z.T) / temperature
#     mask = torch.eye(2*N, device=sim.device).bool()
#     sim = sim.masked_fill(mask, -9e15)
#     pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(sim.device)
#     return nn.CrossEntropyLoss()(sim, pos)

# class Pretrainer:
#     def __init__(self, model, device):
#         self.model = model.to(device)
#         self.device = device

#     def encode(self, input_ids):
#         embedded = self.model.dropout(self.model.embedding(input_ids))
#         output, _ = self.model.rnn(embedded)
#         mean_pool = torch.mean(output, dim=1)
#         max_pool, _ = torch.max(output, dim=1)
#         pooled = torch.cat([mean_pool, max_pool], dim=1)
#         pooled = self.model.ln(pooled)
#         x = self.model.relu(self.model.fc1(self.model.dropout(pooled)))
#         return x

#     def pretrain(self, dataloader, optimizer, epochs=5):
#         self.model.train()
#         for epoch in range(epochs):
#             total_loss = 0
#             for batch in dataloader:
#                 ids1 = batch["input_ids1"].to(self.device)
#                 ids2 = batch["input_ids2"].to(self.device)
#                 z1 = self.encode(ids1)
#                 z2 = self.encode(ids2)
#                 loss = contrastive_loss(z1, z2)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"[Pretrain Epoch {epoch+1}] Contrastive Loss: {total_loss / len(dataloader):.4f}")

# # --- Label-Aware Smoothing ---
# class LabelAwareSmoothingLoss(nn.Module):
#     def __init__(self, class_freqs, smoothing=0.1):
#         super().__init__()
#         self.class_freqs = class_freqs
#         self.smoothing = smoothing
#         self.n_classes = len(class_freqs)
#         self.confidence = 1.0 - smoothing

#     def forward(self, pred, target):
#         with torch.no_grad():
#             true_dist = torch.zeros_like(pred)
#             smoothing_vals = self.smoothing * (1.0 - self.class_freqs[target])
#             true_dist.fill_(self.smoothing / (self.n_classes - 1))
#             true_dist.scatter_(1, target.unsqueeze(1), self.confidence - smoothing_vals.unsqueeze(1))
#         return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))

# # --- Training ---
# class Trainer:
#     def __init__(self, model, device):
#         self.model = model.to(device)
#         self.device = device

#     def train(self, dataloader, optimizer, criterion):
#         self.model.train()
#         total_loss = 0
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(self.device)
#             labels = batch['label'].to(self.device)
#             optimizer.zero_grad()
#             outputs = self.model(input_ids)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#         return total_loss / len(dataloader)

#     def evaluate(self, dataloader, label_encoder, top_n=50):
#         self.model.eval()
#         y_true, y_pred = [], []
#         with torch.no_grad():
#             for batch in dataloader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 labels = batch['label'].to(self.device)
#                 outputs = self.model(input_ids)
#                 preds = outputs.argmax(dim=1)
#                 y_true.extend(labels.cpu().tolist())
#                 y_pred.extend(preds.cpu().tolist())

#         from collections import Counter
#         top_labels = [x for x, _ in Counter(y_true).most_common(top_n)]
#         target_names = [label_encoder.classes_[i] for i in top_labels]
#         report = classification_report(y_true, y_pred, labels=top_labels, target_names=target_names, zero_division=0)
#         f1 = f1_score(y_true, y_pred, average='weighted')
#         return f1, report

# # --- Entry Point ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv_path", default="./clustering_results/papers_with_clusters.csv")
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--pretrain_epochs", type=int, default=1)
#     parser.add_argument("--do_pretrain", action="store_true", default=True)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--embedding_dim", type=int, default=128)
#     parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--max_len", type=int, default=562)
#     parser.add_argument("--vocab_size", type=int, default=20000)
#     parser.add_argument("--save_path", type=str, default="rnn_new_model_long.pt")
#     args = parser.parse_args()

#     df = pd.read_csv(args.csv_path)
#     df["text"] = df["title"].fillna("") + " " + df["cleaned_abstract"].fillna("")
#     df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("") 
#     df = df[df["category"].notna() & df["fine_topic_label"].notna()].copy()
#     df["full_label"] = (
#         df["category"].map(cat_fullname_mapping).apply(clean_label) + "_" +
#         df["fine_topic_label"].apply(clean_label)
#     )
#     df = df[df["full_label"].notna()].copy()

#     le = LabelEncoder()
#     df["label"] = le.fit_transform(df["full_label"])

#     vocab = Vocab(max_size=args.vocab_size)
#     vocab.build(df["text"].tolist())

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = RNNClassifier(
#         vocab_size=len(vocab),
#         embedding_dim=args.embedding_dim,
#         hidden_dim=args.hidden_dim,
#         output_dim=len(le.classes_)
#     )

#     if args.do_pretrain:
#         logging.info("Preparing for self-supervised pretraining...")
#         simclr_dataset = SimCLRTextDataset(df["text"].tolist(), vocab, max_len=args.max_len)
#         simclr_loader = DataLoader(simclr_dataset, batch_size=args.batch_size, shuffle=True)
#         pretrainer = Pretrainer(model, device)
#         pre_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         pretrainer.pretrain(simclr_loader, pre_optimizer, epochs=args.pretrain_epochs)

#     x_train, x_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
#     train_dataset = TextDataset(x_train.tolist(), y_train.tolist(), vocab, max_len=args.max_len)
#     val_dataset = TextDataset(x_val.tolist(), y_val.tolist(), vocab, max_len=args.max_len)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

#     class_counts = np.bincount(y_train, minlength=len(le.classes_))
#     class_freqs = class_counts / class_counts.sum()
#     class_freqs = torch.tensor(class_freqs, dtype=torch.float).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     criterion = LabelAwareSmoothingLoss(class_freqs, smoothing=0.1)
#     trainer = Trainer(model, device)

#     for epoch in range(args.epochs):
#         train_loss = trainer.train(train_loader, optimizer, criterion)
#         f1, report = trainer.evaluate(val_loader, label_encoder=le, top_n=50)
#         print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val F1: {f1:.4f}")
#         print(report)

#     torch.save(model.state_dict(), args.save_path)
#     joblib.dump(le, "encoder_labels_rnn_new_long.pkl")
#     joblib.dump(vocab, "vocab_rnn_model_new_long.pkl")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from constants.constants_map import cat_fullname_mapping
import numpy as np
import random
import argparse
import re
import logging
import string
import joblib
import time
from tqdm import tqdm
import nltk
from torch.nn.functional import normalize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Clean label text ---
def clean_label(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def tokenize_with_math(text):
    # Add spaces around $...$ blocks
    text = re.sub(r"(\$[^$]+\$)", r" \1 ", text)
    # Replace newline/control chars
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return word_tokenize(text.lower())

# --- Scientific Text Augmentation ---
class ScientificTextAugmenter:
    """Enhanced text augmentation for scientific papers."""
    
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
            
        # Scientific domain specific terms
        self.scientific_terms = {
            'analysis': ['examination', 'evaluation', 'assessment', 'study'],
            'method': ['approach', 'technique', 'procedure', 'protocol'],
            'result': ['outcome', 'finding', 'observation', 'measurement'],
            'model': ['framework', 'system', 'structure', 'representation'],
            'data': ['information', 'measurements', 'observations', 'values'],
            'algorithm': ['procedure', 'method', 'technique', 'process'],
            'parameter': ['variable', 'factor', 'coefficient', 'constant'],
            'function': ['operation', 'procedure', 'transformation', 'mapping'],
            'theory': ['principle', 'concept', 'hypothesis', 'proposition'],
            'experiment': ['investigation', 'test', 'trial', 'study']
        }
    
    def is_math_token(self, token):
        """Check if token is a math expression."""
        return token.startswith('$') and token.endswith('$')
    
    def synonym_replacement(self, tokens, p=0.15):
        """Replace scientific terms with domain-specific synonyms."""
        if not tokens:
            return tokens
            
        new_tokens = tokens.copy()
        for i in range(len(new_tokens)):
            if new_tokens[i] in self.scientific_terms and random.random() < p:
                new_tokens[i] = random.choice(self.scientific_terms[new_tokens[i]])
        
        return new_tokens
    
    def random_deletion(self, tokens, p=0.1):
        """Carefully delete tokens while preserving math expressions."""
        if not tokens:
            return tokens
            
        new_tokens = []
        for token in tokens:
            # Don't delete math expressions
            if self.is_math_token(token) or random.random() > p:
                new_tokens.append(token)
                
        # Ensure we don't delete everything
        if not new_tokens:
            return [random.choice(tokens)]
            
        return new_tokens
    
    def random_swap(self, tokens, n=2):
        """Swap nearby tokens to preserve local context."""
        if len(tokens) < 2:
            return tokens
            
        new_tokens = tokens.copy()
        swap_count = min(n, len(tokens) // 4)  # Limit swaps to 25% of tokens
        
        for _ in range(swap_count):
            # Choose a random position
            idx = random.randint(0, len(new_tokens) - 2)
            # Swap with adjacent token
            new_tokens[idx], new_tokens[idx + 1] = new_tokens[idx + 1], new_tokens[idx]
            
        return new_tokens
    
    def augment(self, text, method='weak'):
        """Apply scientific text augmentation with different strengths."""
        tokens = tokenize_with_math(text)
        
        if method == 'strong':
            # Apply multiple augmentations (more aggressive)
            if random.random() < 0.7:
                tokens = self.synonym_replacement(tokens, p=0.2)
            if random.random() < 0.6:
                tokens = self.random_deletion(tokens, p=0.15)
            if random.random() < 0.4:
                tokens = self.random_swap(tokens, n=min(3, len(tokens) // 6))
        elif method == 'medium':
            # Medium strength augmentation
            if random.random() < 0.6:
                tokens = self.synonym_replacement(tokens, p=0.15)
            if random.random() < 0.4:
                tokens = self.random_deletion(tokens, p=0.1)
            if random.random() < 0.3:
                tokens = self.random_swap(tokens, n=min(2, len(tokens) // 8))
        else:  # 'weak'
            # Apply minimal augmentation to preserve scientific meaning
            if random.random() < 0.5:
                tokens = self.synonym_replacement(tokens, p=0.1)
            if random.random() < 0.3:
                tokens = self.random_deletion(tokens, p=0.05)
                
        return " ".join(tokens)

# --- Vocab ---
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

# --- Supervised Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.vocab.encode(self.texts[idx])
        tokens = tokens[:self.max_len] + [0] * max(0, self.max_len - len(tokens))
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- Improved SimCLR Dataset ---
class EnhancedSimCLRTextDataset(Dataset):
    def __init__(self, texts, vocab, max_len=256):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len
        self.augmenter = ScientificTextAugmenter()
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        # Generate two different augmentations
        aug1 = self.augmenter.augment(self.texts[idx], method='weak')
        aug2 = self.augmenter.augment(self.texts[idx], method='medium')
        
        # Encode the texts
        ids1 = self.vocab.encode(aug1)[:self.max_len]
        ids2 = self.vocab.encode(aug2)[:self.max_len]
        
        # Pad sequences
        ids1 += [0] * (self.max_len - len(ids1))
        ids2 += [0] * (self.max_len - len(ids2))
        
        return {
            "input_ids1": torch.tensor(ids1, dtype=torch.long),
            "input_ids2": torch.tensor(ids2, dtype=torch.long)
        }

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

# --- Improved Pretraining helper ---
def nt_xent_loss(z1, z2, temperature=0.1):
    """
    Improved NT-Xent (normalized temperature-scaled cross entropy) loss.
    This is a more stable version of the contrastive loss function.
    """
    # Normalize the representations
    z1 = normalize(z1, dim=1)
    z2 = normalize(z2, dim=1)
    
    # Get batch size
    batch_size = z1.size(0)
    
    # Full batch of representations
    representations = torch.cat([z1, z2], dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(representations, representations.t()) / temperature
    
    # Remove diagonal elements (self-similarity)
    mask = torch.eye(2 * batch_size, device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask) - mask * 1e9
    
    # Create labels for positive pairs
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(similarity_matrix.device)
    
    # Cross entropy loss
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    
    return loss

class ImprovedPretrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def encode(self, input_ids):
        """Get embeddings from the model."""
        # Extract features using the existing RNN architecture
        embedded = self.model.dropout(self.model.embedding(input_ids))
        output, _ = self.model.rnn(embedded)
        
        # Use both mean and max pooling for richer representation
        mean_pool = torch.mean(output, dim=1)
        max_pool, _ = torch.max(output, dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        
        # Apply normalization
        pooled = self.model.ln(pooled)
        
        # Get representation from fully connected layer
        z = self.model.relu(self.model.fc1(self.model.dropout(pooled)))
        
        return z
    
    def pretrain(self, dataloader, optimizer, epochs=5, scheduler=None, grad_clip=1.0):
        """Run improved pretraining with better monitoring and stability."""
        # Track best loss for early stopping
        best_loss = float('inf')
        no_improve = 0
        
        # Train the model
        self.model.train()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            batch_count = 0
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # Get embeddings
                ids1 = batch["input_ids1"].to(self.device)
                ids2 = batch["input_ids2"].to(self.device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Get embeddings
                z1 = self.encode(ids1)
                z2 = self.encode(ids2)
                
                # Compute improved contrastive loss
                loss = nt_xent_loss(z1, z2, temperature=0.1)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                
                # Update weights
                optimizer.step()
                
                # Update learning rate if using scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Update statistics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Calculate epoch statistics
            avg_loss = total_loss / batch_count
            epoch_time = time.time() - epoch_start
            
            # Log results
            logger.info(f"[Pretrain Epoch {epoch+1}/{epochs}] "
                       f"Loss: {avg_loss:.4f} | "
                       f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 2:  # Stop if no improvement for 2 epochs
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info(f"Pretraining completed with best loss: {best_loss:.4f}")
        return best_loss

# --- Create pretraining learning rate scheduler ---
def create_pretrain_scheduler(optimizer, max_steps, warmup_steps=0, min_lr=1e-6):
    """Create a learning rate scheduler with warmup for pretraining."""
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Decay phase
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Label-Aware Smoothing ---
class LabelAwareSmoothingLoss(nn.Module):
    def __init__(self, class_freqs, smoothing=0.1):
        super().__init__()
        self.class_freqs = class_freqs
        self.smoothing = smoothing
        self.n_classes = len(class_freqs)
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            smoothing_vals = self.smoothing * (1.0 - self.class_freqs[target])
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence - smoothing_vals.unsqueeze(1))
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))

# --- Training ---
class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, label_encoder, top_n=50):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        from collections import Counter
        top_labels = [x for x, _ in Counter(y_true).most_common(top_n)]
        target_names = [label_encoder.classes_[i] for i in top_labels]
        report = classification_report(y_true, y_pred, labels=top_labels, target_names=target_names, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1, report

# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./clustering_results/papers_with_clusters.csv")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--do_pretrain", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_lr", type=float, default=2e-3)  # Separate LR for pretraining
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--save_path", type=str, default="rnn_model_new.pt")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["text"] = df["title"].fillna("") + " " + df["cleaned_abstract"].fillna("")
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("") 
    df = df[df["category"].notna() & df["fine_topic_label"].notna()].copy()
    df["full_label"] = (
        df["category"].map(cat_fullname_mapping).apply(clean_label) + "_" +
        df["fine_topic_label"].apply(clean_label)
    )
    df = df[df["full_label"].notna()].copy()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["full_label"])

    vocab = Vocab(max_size=args.vocab_size)
    vocab.build(df["text"].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = RNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(le.classes_)
    )

    if args.do_pretrain:
        logger.info("Preparing for enhanced self-supervised pretraining...")
        # Use the improved dataset for pretraining
        simclr_dataset = EnhancedSimCLRTextDataset(df["text"].tolist(), vocab, max_len=args.max_len)
        simclr_loader = DataLoader(
            simclr_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0 if device.type == 'cuda' else 2
        )
        
        # Initialize improved pretrainer
        pretrainer = ImprovedPretrainer(model, device)
        
        # Use a higher learning rate for pretraining
        pre_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay
        )
        
        # Create learning rate scheduler with warmup
        total_steps = len(simclr_loader) * args.pretrain_epochs
        scheduler = create_pretrain_scheduler(
            pre_optimizer,
            max_steps=total_steps,
            warmup_steps=args.warmup_steps
        )
        
        # Run improved pretraining
        pretrainer.pretrain(
            simclr_loader, 
            pre_optimizer, 
            epochs=args.pretrain_epochs,
            scheduler=scheduler,
            grad_clip=args.grad_clip
        )
        
        logger.info("Pretraining completed. Proceeding to supervised training.")

    x_train, x_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    train_dataset = TextDataset(x_train.tolist(), y_train.tolist(), vocab, max_len=args.max_len)
    val_dataset = TextDataset(x_val.tolist(), y_val.tolist(), vocab, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    class_counts = np.bincount(y_train, minlength=len(le.classes_))
    class_freqs = class_counts / class_counts.sum()
    class_freqs = torch.tensor(class_freqs, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = LabelAwareSmoothingLoss(class_freqs, smoothing=0.1)
    trainer = Trainer(model, device)

    # Track best F1 score for model saving
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_loader, optimizer, criterion)
        f1, report = trainer.evaluate(val_loader, label_encoder=le, top_n=50)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save_path)
            logger.info(f"New best model saved with F1: {f1:.4f}")
            print(report)
        else:
            print(f"No improvement over best F1: {best_f1:.4f}")

    logger.info(f"Training completed. Best F1: {best_f1:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(args.save_path))
    final_f1, final_report = trainer.evaluate(val_loader, label_encoder=le, top_n=50)
    logger.info(f"Final model performance - F1: {final_f1:.4f}")
    print(final_report)
    
    # Save artifacts
    joblib.dump(le, "encoder_labels_rnn_new.pkl")
    joblib.dump(vocab, "vocab_rnn_model_new.pkl")