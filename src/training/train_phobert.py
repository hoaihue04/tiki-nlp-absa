"""
train_phobert.py – PhoBERT Multi-task ASQP với Weighted Loss
=============================================================
Cách chạy:
  cd TIKI
  python src/training/train_phobert.py
"""

import os, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data/training"
MODEL_DIR  = "models/phobert"
LABEL_MAP  = os.path.join(DATA_DIR, "label_map.json")
PHOBERT    = "vinai/phobert-base-v2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 10
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 128
PATIENCE   = 3
SEED       = 42

os.makedirs(MODEL_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"🖥️  Device: {DEVICE.upper()}")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class PhoBERTDataset(Dataset):
    def __init__(self, df, tokenizer, cat2id, sent2id, max_len=MAX_LEN):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cat2id    = cat2id
        self.sent2id   = sent2id
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row     = self.df.iloc[i]
        text    = str(row["text"])
        aspect  = str(row["aspect_term"])
        opinion = str(row["opinion_term"])

        enc = self.tokenizer(
            text,
            f"{aspect} {opinion}",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cat_label":  torch.tensor(
                self.cat2id.get(str(row["aspect_category"]), 0), dtype=torch.long),
            "sent_label": torch.tensor(
                self.sent2id.get(str(row["sentiment"]), 0), dtype=torch.long),
        }


# ─── Model ────────────────────────────────────────────────────────────────────
class PhoBERTASQP(nn.Module):
    def __init__(self, n_categories, n_sentiments,
                 sent_weights=None, cat_weights=None, dropout=0.1):
        """
        sent_weights : Tensor [n_sentiments] — class weights cho sentiment loss
        cat_weights  : Tensor [n_categories] — class weights cho category loss
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(PHOBERT)
        hidden    = self.bert.config.hidden_size    # 768

        self.dropout        = nn.Dropout(dropout)
        self.cat_classifier = nn.Linear(hidden, n_categories)
        self.sen_classifier = nn.Linear(hidden, n_sentiments)

        # ── Loss functions với class weights ──────────────────────────────────
        # weight=None → CrossEntropyLoss thông thường (không dùng weighting)
        # weight=Tensor → nhân hệ số vào loss của từng lớp
        self.loss_cat  = nn.CrossEntropyLoss(
            weight=cat_weights,  # None hoặc Tensor [n_categories]
        )
        self.loss_sent = nn.CrossEntropyLoss(
            weight=sent_weights, # None hoặc Tensor [n_sentiments]
        )

    def forward(self, input_ids, attention_mask,
                cat_labels=None, sent_labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])   # [CLS] token
        cat_logits  = self.cat_classifier(pooled)
        sent_logits = self.sen_classifier(pooled)

        if cat_labels is not None:
            # Weighted loss: mẫu neutral bị phạt nặng hơn khi đoán sai
            loss = (
                self.loss_cat(cat_logits, cat_labels)
                + self.loss_sent(sent_logits, sent_labels)
            )
            return loss, cat_logits, sent_logits

        return cat_logits, sent_logits


# ─── Evaluate ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    cat_true, cat_pred     = [], []
    sent_true, sent_pred   = [], []

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        cl, sl = model(ids, mask)
        cat_pred.extend(cl.argmax(-1).cpu().tolist())
        sent_pred.extend(sl.argmax(-1).cpu().tolist())
        cat_true.extend(batch["cat_label"].tolist())
        sent_true.extend(batch["sent_label"].tolist())

    # Dùng MACRO F1 — không bị ảnh hưởng bởi class imbalance
    cat_f1  = f1_score(cat_true,  cat_pred,  average="macro", zero_division=0)
    sent_f1 = f1_score(sent_true, sent_pred, average="macro", zero_division=0)
    combined = (cat_f1 + sent_f1) / 2

    return combined, cat_f1, sent_f1, cat_true, cat_pred, sent_true, sent_pred


# ─── Train ────────────────────────────────────────────────────────────────────
def train():
    # ── 1. Load label map + class weights ────────────────────────────────────
    with open(LABEL_MAP, encoding="utf-8") as f:
        lmap = json.load(f)

    cats  = lmap["aspect_categories"]
    sents = lmap["sentiments"]
    cat2id  = {c: i for i, c in enumerate(cats)}
    sent2id = {s: i for i, s in enumerate(sents)}
    id2cat  = {i: c for c, i in cat2id.items()}
    id2sent = {i: s for s, i in sent2id.items()}

    print(f"📌 {len(cats)} categories | {len(sents)} sentiments")

    # ── Đọc class weights từ label_map.json ──────────────────────────────────
    sent_w_list = lmap.get("sent_weights_list")
    cat_w_list  = lmap.get("cat_weights_list")

    if sent_w_list and cat_w_list:
        sent_weights = torch.tensor(sent_w_list, dtype=torch.float).to(DEVICE)
        cat_weights  = torch.tensor(cat_w_list,  dtype=torch.float).to(DEVICE)

        print(f"\n⚖️  Sentiment class weights:")
        for i, s in enumerate(sents):
            print(f"     [{i}] {s:<12} → {sent_w_list[i]:.4f}")
        print(f"\n  (neutral được nhân weight {sent_w_list[sents.index('neutral')]:.2f}x "
              f"so với trung bình)")
    else:
        # Fallback: không dùng weighting nếu file cũ chưa có weights
        print("\n⚠️  Không tìm thấy class weights trong label_map.json")
        print("   → Dùng CrossEntropyLoss thông thường")
        print("   → Chạy lại prepare_data.py để tạo weights!")
        sent_weights = None
        cat_weights  = None

    # ── 2. Load dữ liệu ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT)
    loaders   = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"phobert_{split}.csv"))
        print(f"  📂 {split}: {len(df)} quadruples")
        ds = PhoBERTDataset(df, tokenizer, cat2id, sent2id)
        loaders[split] = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(DEVICE == "cuda"),
        )

    # ── 3. Khởi tạo model với weighted loss ──────────────────────────────────
    model = PhoBERTASQP(
        n_categories = len(cats),
        n_sentiments = len(sents),
        sent_weights = sent_weights,   # ← đây là điểm khác biệt chính
        cat_weights  = cat_weights,
    ).to(DEVICE)

    total_steps = len(loaders["train"]) * EPOCHS
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(0.1 * total_steps),
        num_training_steps= total_steps,
    )

    best_f1    = 0.0
    no_improve = 0
    best_path  = os.path.join(MODEL_DIR, "phobert_best.pt")

    print(f"\n🚀 Training PhoBERT ({EPOCHS} epochs | patience={PATIENCE} | "
          f"weighted_loss={'YES' if sent_weights is not None else 'NO'})...")
    print(f"\n{'Epoch':>6} {'Loss':>8} {'Val F1':>8} {'Cat F1':>8} {'Sent F1':>9}")
    print("-" * 48)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in loaders["train"]:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            clbl = batch["cat_label"].to(DEVICE)
            slbl = batch["sent_label"].to(DEVICE)

            loss, _, _ = model(ids, mask, clbl, slbl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loaders["train"])
        val_f1, cat_f1, sent_f1, *_ = evaluate(model, loaders["val"])
        print(f"  {epoch:3d}   {avg_loss:7.4f}   {val_f1:.4f}   "
              f"{cat_f1:.4f}   {sent_f1:.4f}", end="")

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "cat2id":  cat2id,  "id2cat":  id2cat,
                "sent2id": sent2id, "id2sent": id2sent,
                "n_categories": len(cats),
                "n_sentiments": len(sents),
                # Lưu cả weights để load lại khi inference
                "sent_weights": sent_w_list,
                "cat_weights":  cat_w_list,
            }, best_path)
            print("  ✅ best")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"\n  ⏹ Early stopping tại epoch {epoch}")
                break

    # ── 4. Đánh giá test ─────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("📊 TEST SET EVALUATION (best model)")
    print(f"{'='*52}")

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_f1, cat_f1, sent_f1, ct, cp, st, sp = evaluate(model, loaders["test"])

    print(f"  Combined F1  : {test_f1:.4f}")
    print(f"  Category  F1 : {cat_f1:.4f}")
    print(f"  Sentiment F1 : {sent_f1:.4f}")

    print("\n  Sentiment Classification Report:")
    print(classification_report(st, sp, target_names=sents, zero_division=0))

    # Confusion matrix
    print("  Sentiment Confusion Matrix:")
    cm = confusion_matrix(st, sp)
    header = f"  {'':>12}" + "".join(f"{s:>12}" for s in sents)
    print(header)
    for i, row_lbl in enumerate(sents):
        row_str = f"  {row_lbl:>12}" + "".join(f"{cm[i][j]:>12d}" for j in range(len(sents)))
        print(row_str)

    # ── 5. Lưu kết quả ───────────────────────────────────────────────────────
    results = {
        "test_combined_f1": float(test_f1),
        "test_category_f1": float(cat_f1),
        "test_sentiment_f1": float(sent_f1),
        "used_class_weights": sent_weights is not None,
        "sent_weights_used": sent_w_list,
    }
    with open(os.path.join(MODEL_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    print(f"\n✅ Xong! Model → {MODEL_DIR}")


if __name__ == "__main__":
    train()