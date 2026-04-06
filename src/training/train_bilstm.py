"""
train_bilstm.py  (v5 - capture output + auto zip/download)
======================================
Train BiLSTM + CRF cho bài toán ASQP với class-weighted loss.

THAY ĐỔI SO VỚI v4:
  + Import colab_utils (TeeStream, zip_and_download)
  + TeeStream bắt đầu capture NGAY ĐẦU hàm train()
    → toàn bộ stdout/stderr được ghi vào output_<ts>.txt
  + Cuối train(): tee.restore() → zip_and_download()
    → tự động nén models/ + results/ rồi tải về máy

Cách chạy:
  cd TIKI
  pip install torch transformers seqeval
  python src/training/train_bilstm.py
"""

import os, json, random, logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import f1_score as sk_f1, classification_report as sk_report

# ▼▼▼ THÊM MỚI ▼▼▼
from src.training.colab_utils import TeeStream, zip_and_download
# ▲▲▲ THÊM MỚI ▲▲▲

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = "data/training"
MODEL_DIR   = "models/bilstm"
LOG_DIR     = "results/logs"
LABEL_MAP   = os.path.join(DATA_DIR, "label_map.json")
PHOBERT     = "vinai/phobert-base-v2"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 15
BATCH_SIZE  = 16
LR          = 1e-3
MAX_LEN     = 128
SEED        = 42
PATIENCE    = 4

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# ════════════════════════════════════════════════════════════════════════════
# 0. LOGGER SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: str) -> tuple[logging.Logger, str, str]:
    """
    Khởi tạo logger ghi đồng thời ra console và file .log.
    Trả về (logger, đường dẫn file log, timestamp dùng cho các file khác).
    """
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"bilstm_{ts}.log")
    hist_path = os.path.join(log_dir, f"bilstm_{ts}_history.json")

    logger = logging.getLogger("bilstm_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path, hist_path, ts


# ════════════════════════════════════════════════════════════════════════════
# 1. ĐỌC FILE BIO
# ════════════════════════════════════════════════════════════════════════════

def read_bio_file(path):
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split("\t")
                if len(parts) == 4:
                    current.append(tuple(parts))
    if current:
        sentences.append(current)
    return sentences


# ════════════════════════════════════════════════════════════════════════════
# 2. ALIGN WORD → SUBTOKEN
# ════════════════════════════════════════════════════════════════════════════

def align_labels_with_subtokens(words, labels, tokenizer, max_len, bio2id):
    CLS_ID = tokenizer.cls_token_id
    SEP_ID = tokenizer.sep_token_id
    PAD_ID = tokenizer.pad_token_id

    subtoken_ids  = []
    label_ids_raw = []

    for word, lbl in zip(words, labels):
        sub = tokenizer.tokenize(word) or [tokenizer.unk_token]
        ids = tokenizer.convert_tokens_to_ids(sub)
        subtoken_ids.append(ids[0])
        label_ids_raw.append(bio2id.get(lbl, bio2id["O"]))
        for sid in ids[1:]:
            subtoken_ids.append(sid)
            label_ids_raw.append(-100)

    max_body      = max_len - 2
    subtoken_ids  = subtoken_ids[:max_body]
    label_ids_raw = label_ids_raw[:max_body]

    input_ids  = [CLS_ID] + subtoken_ids + [SEP_ID]
    label_ids  = [-100]   + label_ids_raw + [-100]
    seq_len    = len(input_ids)
    attn_mask  = [1] * seq_len
    pad_len    = max_len - seq_len

    input_ids  += [PAD_ID] * pad_len
    attn_mask  += [0]      * pad_len
    label_ids  += [-100]   * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
        torch.tensor(label_ids, dtype=torch.long),
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ════════════════════════════════════════════════════════════════════════════

class BIODataset(Dataset):
    def __init__(self, sentences, tokenizer, bio2id, sent2id, max_len=MAX_LEN):
        self.samples = []
        skipped = 0
        for sent in sentences:
            if not sent:
                continue
            words    = [r[0] for r in sent]
            bio_lbls = [r[1] for r in sent]
            sent_lbl = sent[0][2] if len(sent[0]) > 2 else "positive"
            try:
                ids, mask, lbl = align_labels_with_subtokens(
                    words, bio_lbls, tokenizer, max_len, bio2id
                )
                self.samples.append({
                    "input_ids":      ids,
                    "attention_mask": mask,
                    "bio_labels":     lbl,
                    "sent_label": torch.tensor(
                        sent2id.get(sent_lbl, 0), dtype=torch.long
                    ),
                })
            except Exception:
                skipped += 1
        if skipped:
            print(f"  ⚠️ Bỏ qua {skipped} câu lỗi tokenize")

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ════════════════════════════════════════════════════════════════════════════
# 4. CRF
# ════════════════════════════════════════════════════════════════════════════

class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags          = num_tags
        self.transitions       = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions   = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions,       -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions,   -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        score     = self._score_sentence(emissions, tags, mask)
        partition = self._forward_alg(emissions, mask)
        return (partition - score).mean()

    def _score_sentence(self, emissions, tags, mask):
        B, L, C = emissions.shape
        score   = self.start_transitions[tags[:, 0]]
        score  += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, L):
            m     = mask[:, t].float()
            trans = self.transitions[tags[:, t], tags[:, t-1]]
            emit  = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            score += (trans + emit) * m
        seq_ends  = mask.long().sum(1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score    += self.end_transitions[last_tags]
        return score

    def _forward_alg(self, emissions, mask):
        B, L, C = emissions.shape
        alpha   = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, L):
            m         = mask[:, t].unsqueeze(1)
            scores    = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            new_alpha = torch.logsumexp(scores, dim=1) + emissions[:, t]
            alpha     = torch.where(m.bool(), new_alpha, alpha)
        alpha += self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        B, L, C     = emissions.shape
        viterbi     = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for t in range(1, L):
            scores      = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=1)
            new_vit     = best_scores + emissions[:, t]
            m           = mask[:, t].unsqueeze(1)
            viterbi     = torch.where(m.bool(), new_vit, viterbi)
            backpointers.append(best_tags)
        viterbi  += self.end_transitions.unsqueeze(0)
        best_last = viterbi.argmax(dim=1)
        paths     = []
        for b in range(B):
            path = [best_last[b].item()]
            for bp in reversed(backpointers):
                path.append(bp[b, path[-1]].item())
            path.reverse()
            seq_len = int(mask[b].long().sum().item())
            paths.append(path[:seq_len])
        return paths


# ════════════════════════════════════════════════════════════════════════════
# 5. MODEL
# ════════════════════════════════════════════════════════════════════════════

class BiLSTMCRF(nn.Module):
    def __init__(self, n_bio_labels, n_sentiments,
                 sent_weights=None, bio_weights=None,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(PHOBERT, add_pooling_layer=False)
        for p in self.bert.parameters():
            p.requires_grad = False

        bert_dim = self.bert.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size    = bert_dim,
            hidden_size   = hidden_dim,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc_bio   = nn.Linear(hidden_dim * 2, n_bio_labels)
        self.crf      = CRF(n_bio_labels)
        self.fc_sent  = nn.Linear(bert_dim, n_sentiments)
        self.loss_sent = nn.CrossEntropyLoss(weight=sent_weights)

    def _encode(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        return out.last_hidden_state

    def forward(self, input_ids, attention_mask,
                bio_labels=None, sent_labels=None):
        emb         = self._encode(input_ids, attention_mask)
        lstm_out, _ = self.bilstm(self.dropout(emb))
        emissions   = self.fc_bio(self.dropout(lstm_out))
        cls_emb     = self.dropout(emb[:, 0, :])
        sent_logits = self.fc_sent(cls_emb)
        mask        = attention_mask.bool()

        if bio_labels is not None and sent_labels is not None:
            crf_labels = bio_labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss_bio  = self.crf(emissions, crf_labels, mask)
            loss_sent = self.loss_sent(sent_logits, sent_labels)
            return loss_bio + loss_sent, sent_logits

        bio_preds  = self.crf.decode(emissions, mask)
        sent_preds = sent_logits.argmax(dim=-1)
        return bio_preds, sent_preds


# ════════════════════════════════════════════════════════════════════════════
# 6. EVALUATE
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, id2bio, id2sent, sent_labels_list):
    model.eval()
    all_bio_true, all_bio_pred   = [], []
    all_sent_true, all_sent_pred = [], []

    with torch.no_grad():
        for batch in loader:
            ids       = batch["input_ids"].to(DEVICE)
            mask      = batch["attention_mask"].to(DEVICE)
            bio_lbls  = batch["bio_labels"]
            sent_lbls = batch["sent_label"].tolist()

            bio_preds, sent_preds = model(ids, mask)

            for pred_seq, true_seq in zip(bio_preds, bio_lbls.tolist()):
                tt, tp = [], []
                for pos, tid in enumerate(true_seq):
                    if tid == -100:
                        continue
                    tt.append(id2bio.get(tid, "O"))
                    pid = pred_seq[pos] if pos < len(pred_seq) else 0
                    tp.append(id2bio.get(pid, "O"))
                if tt:
                    all_bio_true.append(tt)
                    all_bio_pred.append(tp)

            all_sent_true.extend(sent_lbls)
            all_sent_pred.extend(sent_preds.cpu().tolist())

    bio_f1  = seq_f1(all_bio_true, all_bio_pred,
                     average="micro", zero_division=0) if all_bio_true else 0.0
    sent_f1 = sk_f1(all_sent_true, all_sent_pred,
                    average="macro", zero_division=0)
    combined = (bio_f1 + sent_f1) / 2

    return combined, bio_f1, sent_f1, \
           all_bio_true, all_bio_pred, \
           all_sent_true, all_sent_pred


# ════════════════════════════════════════════════════════════════════════════
# 7. TRAIN
# ════════════════════════════════════════════════════════════════════════════

def train():
    # ▼▼▼ THÊM MỚI: bắt đầu capture stdout/stderr vào file output_<ts>.txt ▼▼▼
    # Phải gọi TRƯỚC setup_logger để capture cả log output lẫn print() thường
    _pre_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeStream(log_dir=LOG_DIR, ts=_pre_ts)
    # ▲▲▲ THÊM MỚI ▲▲▲

    # ── Logger ────────────────────────────────────────────────────────────────
    logger, log_path, hist_path, ts = setup_logger(LOG_DIR)
    epoch_history = []

    logger.info("=" * 72)
    logger.info(f"BiLSTM+CRF Training  —  run: {ts}")
    logger.info(f"Device : {DEVICE}  |  Epochs: {EPOCHS}  |  "
                f"Batch: {BATCH_SIZE}  |  LR: {LR}")
    logger.info(f"Log    : {log_path}")
    logger.info("=" * 72)

    # ── Label map + class weights ─────────────────────────────────────────────
    with open(LABEL_MAP, encoding="utf-8") as f:
        lmap = json.load(f)

    bio_labels = lmap["bio_labels"]
    sents      = lmap["sentiments"]
    bio2id     = {l: i for i, l in enumerate(bio_labels)}
    id2bio     = {i: l for l, i in bio2id.items()}
    sent2id    = {s: i for i, s in enumerate(sents)}
    id2sent    = {i: s for s, i in sent2id.items()}

    sent_w_list = lmap.get("sent_weights_list")
    if sent_w_list:
        sent_weights = torch.tensor(sent_w_list, dtype=torch.float).to(DEVICE)
        logger.info("⚖️  Sentiment class weights:")
        for i, s in enumerate(sents):
            logger.info(f"     [{i}] {s:<12} → {sent_w_list[i]:.4f}")
    else:
        logger.warning("⚠️  Không có class weights → dùng CrossEntropyLoss thông thường")
        logger.warning("   Chạy lại prepare_data.py để tạo weights!")
        sent_weights = None

    logger.info(f"📌 BIO labels ({len(bio_labels)}): {bio_labels}")
    logger.info(f"📌 Sentiments ({len(sents)}): {sents}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("\n🔤 Load tokenizer PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT, use_fast=False)

    # ── Dữ liệu ──────────────────────────────────────────────────────────────
    logger.info("\n📂 Đọc dữ liệu BIO...")
    train_sents = read_bio_file(os.path.join(DATA_DIR, "bilstm_train.txt"))
    val_sents   = read_bio_file(os.path.join(DATA_DIR, "bilstm_val.txt"))
    test_sents  = read_bio_file(os.path.join(DATA_DIR, "bilstm_test.txt"))
    logger.info(f"   train={len(train_sents)}, val={len(val_sents)}, "
                f"test={len(test_sents)}")

    logger.info("\n⚙️  Tokenize & align labels...")
    train_ds = BIODataset(train_sents, tokenizer, bio2id, sent2id)
    val_ds   = BIODataset(val_sents,   tokenizer, bio2id, sent2id)
    test_ds  = BIODataset(test_sents,  tokenizer, bio2id, sent2id)
    logger.info(f"   Samples: train={len(train_ds)}, val={len(val_ds)}, "
                f"test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"\n🔨 Khởi tạo BiLSTM+CRF (PhoBERT frozen, weighted_loss="
                f"{'YES' if sent_weights is not None else 'NO'})...")
    model = BiLSTMCRF(
        n_bio_labels = len(bio_labels),
        n_sentiments = len(sents),
        sent_weights = sent_weights,
    ).to(DEVICE)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Tổng params   : {n_total:,}")
    logger.info(f"   Trainable     : {n_trainable:,} ({n_trainable/n_total*100:.1f}%)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_f1    = 0.0
    no_improve = 0
    best_path  = os.path.join(MODEL_DIR, "bilstm_best.pt")

    header = (f"{'Ep':>4} {'Loss':>8} {'Val F1':>8} {'BIO F1':>8} "
              f"{'Sent F1':>9} {'Best':>8} {'Note'}")
    sep    = "-" * 72
    logger.info(f"\n🚀 Training ({EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR})")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            ids       = batch["input_ids"].to(DEVICE)
            mask      = batch["attention_mask"].to(DEVICE)
            bio_lbls  = batch["bio_labels"].to(DEVICE)
            sent_lbls = batch["sent_label"].to(DEVICE)

            loss, _ = model(ids, mask, bio_lbls, sent_lbls)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        combined, bio_f1, sent_f1, *_ = evaluate(
            model, val_loader, id2bio, id2sent, sents
        )
        scheduler.step(combined)

        note = ""
        if combined > best_f1:
            best_f1    = combined
            no_improve = 0
            note       = "✅ saved"
            torch.save({
                "model_state": model.state_dict(),
                "bio2id":      bio2id,
                "id2bio":      id2bio,
                "sent2id":     sent2id,
                "id2sent":     id2sent,
                "n_bio_labels":  len(bio_labels),
                "n_sentiments":  len(sents),
                "hidden_dim":    256,
                "sent_weights":  sent_w_list,
            }, best_path)
        else:
            no_improve += 1

        epoch_record = {
            "epoch":        epoch,
            "train_loss":   round(avg_loss,  4),
            "val_combined": round(combined,  4),
            "val_bio_f1":   round(bio_f1,    4),
            "val_sent_f1":  round(sent_f1,   4),
            "best_f1":      round(best_f1,   4),
            "saved":        note != "",
            "lr":           round(optimizer.param_groups[0]["lr"], 6),
        }
        epoch_history.append(epoch_record)

        logger.info(
            f"  {epoch:3d}  {avg_loss:8.4f}  {combined:8.4f}  "
            f"{bio_f1:8.4f}  {sent_f1:9.4f}  {best_f1:8.4f}  {note}"
        )

        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(epoch_history, f, ensure_ascii=False, indent=2)

        if no_improve >= PATIENCE:
            logger.info(f"  ⏹️  Early stopping tại epoch {epoch} "
                        f"(no_improve={no_improve})")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*52}")
    logger.info("📊 ĐÁNH GIÁ TEST (best model)")
    logger.info(f"{'='*52}")

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    combined, bio_f1, sent_f1, \
        bio_true, bio_pred, \
        sent_true, sent_pred = evaluate(
            model, test_loader, id2bio, id2sent, sents
        )

    logger.info(f"  Combined F1  : {combined:.4f}")
    logger.info(f"  BIO F1       : {bio_f1:.4f}")
    logger.info(f"  Sentiment F1 : {sent_f1:.4f}")

    if bio_true:
        bio_report = seq_report(bio_true, bio_pred, zero_division=0)
        logger.info("\n  BIO Report:\n" + bio_report)

    sent_report = sk_report(sent_true, sent_pred,
                            target_names=sents, zero_division=0)
    logger.info("\n  Sentiment Report:\n" + sent_report)

    results = {
        "run_timestamp":       ts,
        "log_file":            log_path,
        "history_file":        hist_path,
        "config": {
            "epochs":     EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr":         LR,
            "max_len":    MAX_LEN,
            "patience":   PATIENCE,
            "seed":       SEED,
            "device":     DEVICE,
        },
        "test_combined_f1":    float(combined),
        "test_bio_f1":         float(bio_f1),
        "test_sentiment_f1":   float(sent_f1),
        "used_class_weights":  sent_weights is not None,
        "sent_weights_used":   sent_w_list,
        "epoch_history":       epoch_history,
    }
    results_path = os.path.join(MODEL_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

    logger.info(f"\n✅ Xong!")
    logger.info(f"   Model    : {best_path}")
    logger.info(f"   Results  : {results_path}")
    logger.info(f"   Log text : {log_path}")
    logger.info(f"   History  : {hist_path}")

    # ▼▼▼ THÊM MỚI: khôi phục output, rồi zip & download về máy ▼▼▼
    tee.restore()   # đóng file output_<ts>.txt; stdout/stderr trở về bình thường

    zip_and_download(
        dirs_to_zip = [MODEL_DIR, LOG_DIR],   # nén models/bilstm + results/logs
        extra_files = [],                      # file lẻ bổ sung nếu cần
        output_name = f"bilstm_run_{ts}.zip",  # tên zip dễ nhận biết theo run
    )
    # ▲▲▲ THÊM MỚI ▲▲▲


if __name__ == "__main__":
    train()