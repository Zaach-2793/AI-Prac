import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import joblib
from constants.constants_map import cat_fullname_mapping

class BertTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, cat_labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.cat_labels = cat_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        cat_label = self.cat_labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'cat_label': torch.tensor(cat_label, dtype=torch.long)
        }

class HierarchicalRobertaClassifier(nn.Module):
    def __init__(self, num_categories, num_fine_labels, dropout_rate=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.encoder.config.hidden_size

        self.category_classifier = nn.Linear(hidden_size, num_categories)
        self.fine_classifier = nn.Sequential(
            nn.Linear(hidden_size + num_categories, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_fine_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        mean_pooled = sum_hidden / sum_mask

        cat_logits = self.category_classifier(self.dropout(mean_pooled))
        cat_probs = torch.softmax(cat_logits, dim=1)
        concat = torch.cat([mean_pooled, cat_probs], dim=1)
        fine_logits = self.fine_classifier(concat)
        return cat_logits, fine_logits

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion_cat, criterion_fine, scheduler):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            fine_labels = batch['label'].to(self.device)
            cat_labels = batch['cat_label'].to(self.device)

            optimizer.zero_grad()
            cat_logits, fine_logits = self.model(input_ids, attention_mask)
            loss = criterion_cat(cat_logits, cat_labels) + criterion_fine(fine_logits, fine_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, label_encoder, top_n=None):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                fine_labels = batch['label'].to(self.device)
                _, fine_logits = self.model(input_ids, attention_mask)
                preds = fine_logits.argmax(dim=1)
                y_true.extend(fine_labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        from collections import Counter
        if top_n is not None:
            label_counter = Counter(y_true)
            top_labels = [label for label, _ in label_counter.most_common(top_n)]
            target_names = label_encoder.inverse_transform(top_labels)
            report = classification_report(
                y_true, y_pred,
                labels=top_labels,
                target_names=target_names,
                zero_division=0
            )
            f1 = f1_score(y_true, y_pred, labels=top_labels, average='weighted')
        else:
            all_labels = np.arange(len(label_encoder.classes_))
            report = classification_report(
                y_true, y_pred,
                labels=all_labels,
                target_names=label_encoder.classes_,
                zero_division=0
            )
            f1 = f1_score(y_true, y_pred, labels=all_labels, average='weighted')

        return f1, report
      
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./clustering_results/papers_with_clusters.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_path", type=str, default="hier_roberta_model.pt")
    parser.add_argument("--encoder_path", type=str, default="label_encoder_bert.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df["text"] = df["title"].fillna("") + " " + df["cleaned_abstract"].fillna("")
    df = df[df["category"].notna() & df["fine_topic_label"].notna()].copy()
    df["category_label"] = df["category"].str.lower().str.replace(" ", "_")
    df["full_label"] = (
        df["category"].map(cat_fullname_mapping).str.lower().str.replace(" ", "_") + "_" +
        df["fine_topic_label"].str.lower().str.replace(" ", "_")
    )
    df = df[df["full_label"].notna()].copy()

    cat_le = LabelEncoder()
    fine_le = LabelEncoder()
    df["cat_label"] = cat_le.fit_transform(df["category_label"])
    df["label_id"] = fine_le.fit_transform(df["full_label"])
    joblib.dump(fine_le, args.encoder_path)

    x_train, x_val, y_train, y_val = train_test_split(df["text"], df["label_id"], test_size=0.2, random_state=42)
    cat_train = df.loc[x_train.index, "cat_label"].tolist()
    cat_val = df.loc[x_val.index, "cat_label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataset = BertTextDataset(x_train.tolist(), y_train.tolist(), tokenizer, cat_train)
    val_dataset = BertTextDataset(x_val.tolist(), y_val.tolist(), tokenizer, cat_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = HierarchicalRobertaClassifier(
        num_categories=len(cat_le.classes_),
        num_fine_labels=len(fine_le.classes_)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device)

    from collections import Counter
    label_counts = Counter(y_train)
    num_labels = len(fine_le.classes_)
    weights = []
    total_samples = len(y_train)
    for i in range(num_labels):
        count = label_counts.get(i, 0)
        weight = total_samples / (count * num_labels) if count > 0 else 0.0
        weights.append(weight)

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    criterion_cat = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader,optimizer,criterion_cat, criterion_fine, scheduler)
        f1, report = trainer.evaluate(val_loader, label_encoder=fine_le, top_n=50)
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {train_loss:.4f}, F1 Score = {f1:.4f}")
        print(report)

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
