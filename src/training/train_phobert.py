#!/usr/bin/env python3
"""
train_phobert.py — PhoBERT Multi-task cho ABSA (AD + AP)
=========================================================
Kiến trúc (theo paper VLSP 2018):
    PhoBERT-base → Concat last 4 hidden layers → Linear(3072→768) → 17 heads

Chiến lược tốt nhất từ bài báo:
    - Concat last 4 BERT layers → F1 cao nhất (96.1% dev)
    - Multi-task: 17 heads, mỗi head 4-class (none/pos/neu/neg)
    - Focal Loss (γ=2) cho mất cân bằng, KHÔNG dùng class weights

Checkpoint / Resume (Colab):
    - Lưu sau mỗi epoch vào checkpoints/phobert/epoch_{n}.pt
    - Best model theo avg(AD-F1 + AP-F1)
    - Resume tự động từ checkpoint mới nhất

Cách chạy trên Colab:
    !pip install transformers sentencepiece
    !python src/training/train_phobert.py
    # Nếu mất kết nối → chạy lại, sẽ tự resume

Phụ thuộc:
    pip install torch transformers sentencepiece scikit-learn numpy pandas
"""

import json
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Config ────────────────────────────────────────────────────────────────
class Config:
    # Paths
    DATA_DIR       = 'data/training'
    CHECKPOINT_DIR = 'checkpoints/phobert'
    MODEL_DIR      = 'models/phobert'

    # PhoBERT
    PHOBERT_NAME   = 'vinai/phobert-base-v2'  # PhoBERT v2 (tốt hơn v1)
    MAX_LEN        = 256
    LAST_N_LAYERS  = 4    # concat 4 lớp cuối cùng → 768*4=3072

    # Model head
    PROJ_DIM       = 768  # sau khi project từ 3072 → 768
    DROPOUT        = 0.1

    # Training
    BATCH_SIZE     = 32
    EPOCHS         = 15
    LR             = 2e-5
    LR_HEAD        = 1e-4  # learning rate riêng cho classification heads
    WEIGHT_DECAY   = 1e-2
    GRAD_CLIP      = 1.0
    WARMUP_RATIO   = 0.10  # 10% total steps cho warmup
    PATIENCE       = 5
    SAVE_EVERY     = 1

    # Loss
    FOCAL_GAMMA    = 2.0

    # Model structure
    NUM_CATS       = 17
    NUM_SENT       = 4     # none/positive/neutral/negative


CFG = Config()
SENT_NAMES = ['none', 'positive', 'neutral', 'negative']


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════
class PhoBERTDataset(Dataset):
    """
    Đọc CSV: text, label_0, ..., label_16
    Tokenize bằng PhoBERT tokenizer.
    """
    def __init__(self, csv_path: str, tokenizer, max_len: int = CFG.MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        df             = pd.read_csv(csv_path, dtype=str)
        self.texts     = df['text'].fillna('').tolist()
        label_cols     = [f'label_{i}' for i in range(CFG.NUM_CATS)]
        self.labels    = df[label_cols].fillna(0).astype(int).values  # [N, 17]
        print(f'    Loaded {len(self.texts):,} samples từ {csv_path}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc  = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════
class PhoBERTASQP(nn.Module):
    """
    PhoBERT + concat last 4 layers + 17 classification heads.
    Theo phương pháp tốt nhất trong bài báo VLSP 2018.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg        = cfg
        self.bert       = AutoModel.from_pretrained(
            cfg.PHOBERT_NAME,
            output_hidden_states=True,  # cần để lấy tất cả hidden layers
        )
        bert_hidden     = 768
        concat_dim      = bert_hidden * cfg.LAST_N_LAYERS  # 768*4=3072

        # Projection layer: 3072 → 768
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, cfg.PROJ_DIM),
            nn.LayerNorm(cfg.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )

        # 17 classification heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.PROJ_DIM, 128),
                nn.GELU(),
                nn.Dropout(cfg.DROPOUT),
                nn.Linear(128, cfg.NUM_SENT),
            )
            for _ in range(cfg.NUM_CATS)
        ])

    def forward(self, input_ids, attention_mask):
        """
        input_ids, attention_mask: [B, seq_len]
        → cls_logits: list of 17 tensors [B, 4]
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Lấy last N hidden states: tuple of [B, seq, 768]
        # outputs.hidden_states: (embedding, layer1, ..., layer12)
        # Lấy 4 lớp cuối: layers 9,10,11,12 (index -4,-3,-2,-1)
        hidden_states = outputs.hidden_states  # tuple len=13
        last_n = hidden_states[-(self.cfg.LAST_N_LAYERS):]  # 4 tensors [B, S, 768]

        # Lấy [CLS] token từ mỗi lớp
        cls_vectors = [h[:, 0, :] for h in last_n]           # 4 × [B, 768]
        concat      = torch.cat(cls_vectors, dim=-1)          # [B, 3072]

        # Project
        proj = self.projection(concat)  # [B, 768]

        # Classify
        logits = [cls(proj) for cls in self.classifiers]  # 17 × [B, 4]
        return logits


# ══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()


focal_loss_fn = FocalLoss(gamma=CFG.FOCAL_GAMMA)


# ══════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids  = batch['input_ids'].to(device)
            attn_mask  = batch['attention_mask'].to(device)
            labels     = batch['labels'].cpu().numpy()

            cls_logits = model(input_ids, attn_mask)
            preds = np.stack(
                [l.argmax(dim=-1).cpu().numpy() for l in cls_logits], axis=1)

            all_true.append(labels)
            all_pred.append(preds)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return compute_ad_ap_f1(y_true, y_pred)


def compute_ad_ap_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # AD: binary detection
    y_true_ad = (y_true != 0).astype(int).flatten()
    y_pred_ad = (y_pred != 0).astype(int).flatten()
    ad_p  = precision_score(y_true_ad, y_pred_ad, zero_division=0)
    ad_r  = recall_score(y_true_ad, y_pred_ad, zero_division=0)
    ad_f1 = f1_score(y_true_ad, y_pred_ad, zero_division=0)

    # AP: sentiment cho aspect hiện diện
    mask      = (y_true != 0)
    y_true_ap = y_true[mask]
    y_pred_ap = y_pred[mask]
    if len(y_true_ap) == 0:
        ap_p = ap_r = ap_f1 = 0.0
    else:
        ap_p  = precision_score(y_true_ap, y_pred_ap,
                                labels=[1, 2, 3], average='micro', zero_division=0)
        ap_r  = recall_score(y_true_ap, y_pred_ap,
                             labels=[1, 2, 3], average='micro', zero_division=0)
        ap_f1 = f1_score(y_true_ap, y_pred_ap,
                         labels=[1, 2, 3], average='micro', zero_division=0)

    return {
        'ad_precision': ad_p,
        'ad_recall':    ad_r,
        'ad_f1':        ad_f1,
        'ap_precision': ap_p,
        'ap_recall':    ap_r,
        'ap_f1':        ap_f1,
        'avg_f1':       (ad_f1 + ap_f1) / 2,
    }


# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_latest_checkpoint(checkpoint_dir: str):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    files = [f for f in os.listdir(checkpoint_dir)
             if f.startswith('epoch_') and f.endswith('.pt')]
    if not files:
        return None, 0
    # Thử từ epoch mới nhất xuống — bỏ qua file corrupt (EOFError/pickle lỗi)
    epochs = sorted([int(f.replace('epoch_', '').replace('.pt', ''))
                     for f in files], reverse=True)
    for ep in epochs:
        path = os.path.join(checkpoint_dir, f'epoch_{ep}.pt')
        try:
            state = torch.load(path, map_location='cpu')
            print(f'  [RESUME] Checkpoint epoch {ep}: {path}')
            return state, ep
        except Exception as e:
            print(f'  [!] epoch_{ep}.pt bị corrupt ({type(e).__name__}), bỏ qua...')
            os.remove(path)  # xóa file hỏng để tránh lần sau lại gặp
    print('  [!] Tất cả checkpoint đều corrupt. Train từ đầu.')
    return None, 0


def _cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Giữ lại keep_last checkpoint mới nhất, xóa các epoch cũ hơn."""
    files = [f for f in os.listdir(checkpoint_dir)
             if f.startswith('epoch_') and f.endswith('.pt')]
    if len(files) <= keep_last:
        return
    epochs = sorted([int(f.replace('epoch_', '').replace('.pt', ''))
                     for f in files])
    for ep in epochs[:-keep_last]:
        p = os.path.join(checkpoint_dir, f'epoch_{ep}.pt')
        if os.path.exists(p):
            os.remove(p)


def sync_to_drive(local_dir: str, drive_dir: str):
    """Sync local dir to Google Drive (nếu đang chạy trong Colab)."""
    try:
        import shutil
        if not os.path.exists('/content/drive/MyDrive'):
            return
        os.makedirs(drive_dir, exist_ok=True)
        for item in os.listdir(local_dir):
            src = os.path.join(local_dir, item)
            dst = os.path.join(drive_dir, item)
            if os.path.isfile(src):
                if (not os.path.exists(dst) or
                        os.path.getmtime(src) > os.path.getmtime(dst)):
                    shutil.copy2(src, dst)
        print(f'  [Drive] Synced {local_dir} -> {drive_dir}')
    except Exception as e:
        print(f'  [Drive] Sync skipped: {e}')


# ══════════════════════════════════════════════════════════════════════════
# TRAIN EPOCH
# ══════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids  = batch['input_ids'].to(device)
        attn_mask  = batch['attention_mask'].to(device)
        labels     = batch['labels'].to(device)

        cls_logits = model(input_ids, attn_mask)

        loss = torch.tensor(0.0, device=device)
        for k, logits_k in enumerate(cls_logits):
            loss = loss + focal_loss_fn(logits_k, labels[:, k])
        loss = loss / len(cls_logits)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[PhoBERT] Device: {device}')
    if device.type == 'cuda':
        print(f'          GPU: {torch.cuda.get_device_name(0)}')

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    # 1. Tokenizer
    print(f'\n[1] Tải tokenizer: {CFG.PHOBERT_NAME}...')
    tokenizer = AutoTokenizer.from_pretrained(CFG.PHOBERT_NAME)

    # 2. Dataset
    print('\n[2] Tải dataset...')
    train_ds = PhoBERTDataset(f'{CFG.DATA_DIR}/phobert_train.csv', tokenizer)
    val_ds   = PhoBERTDataset(f'{CFG.DATA_DIR}/phobert_val.csv',   tokenizer)
    test_ds  = PhoBERTDataset(f'{CFG.DATA_DIR}/phobert_test.csv',  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              shuffle=True,  num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE * 2,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.BATCH_SIZE * 2,
                              shuffle=False, num_workers=0)

    # 3. Model
    print(f'\n[3] Khởi tạo PhoBERT ASQP model...')
    model = PhoBERTASQP(CFG).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Total params: {total_p:,} | Trainable: {train_p:,}')

    # 4. Optimizer — learning rate khác nhau cho BERT và heads
    bert_params  = list(model.bert.parameters())
    head_params  = list(model.projection.parameters()) + \
                   list(model.classifiers.parameters())
    optimizer = AdamW([
        {'params': bert_params,  'lr': CFG.LR,      'weight_decay': CFG.WEIGHT_DECAY},
        {'params': head_params,  'lr': CFG.LR_HEAD,  'weight_decay': CFG.WEIGHT_DECAY * 0.1},
    ])

    # Scheduler: linear warmup → cosine decay
    total_steps  = len(train_loader) * CFG.EPOCHS
    warmup_steps = int(total_steps * CFG.WARMUP_RATIO)
    warmup_sched = LinearLR(optimizer, start_factor=0.05, end_factor=1.0,
                            total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=total_steps - warmup_steps,
                                     eta_min=1e-6)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_sched, cosine_sched],
                             milestones=[warmup_steps])

    # 5. Resume
    start_epoch  = 1
    best_avg_f1  = 0.0
    no_improve   = 0
    best_metrics = {}

    ckpt, last_ep = load_latest_checkpoint(CFG.CHECKPOINT_DIR)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state'])
        # Khong load optimizer/scheduler state — incompatible giua cac PyTorch versions
        # Model weights la du de resume dung, optimizer se bat dau lai tu dau
        start_epoch  = last_ep + 1
        best_avg_f1  = ckpt.get('best_avg_f1', 0.0)
        no_improve   = ckpt.get('no_improve', 0)
        best_metrics = ckpt.get('best_metrics', {})
        print(f'  Resumed epoch {last_ep}, best_avg_f1={best_avg_f1:.4f}')

    # 6. Training
    print(f'\n[4] Training từ epoch {start_epoch}/{CFG.EPOCHS}')
    print('-' * 65)

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_m      = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        avg_f1 = val_m['avg_f1']
        print(f'Epoch {epoch:02d}/{CFG.EPOCHS} | '
              f'loss={train_loss:.4f} | '
              f'AD-F1={val_m["ad_f1"]:.4f} | '
              f'AP-F1={val_m["ap_f1"]:.4f} | '
              f'avg={avg_f1:.4f} | '
              f't={elapsed:.1f}s')

        # Checkpoint
        ckpt_data = {
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_avg_f1':     best_avg_f1,
            'no_improve':      no_improve,
            'best_metrics':    best_metrics,
            'val_metrics':     val_m,
        }
        save_checkpoint(ckpt_data, f'{CFG.CHECKPOINT_DIR}/epoch_{epoch}.pt')

        # Xóa checkpoint cũ (giữ 3 gần nhất để tiết kiệm dung lượng)
        _cleanup_checkpoints(CFG.CHECKPOINT_DIR, keep_last=3)

        # Best model
        if avg_f1 > best_avg_f1:
            best_avg_f1  = avg_f1
            best_metrics = val_m.copy()
            no_improve   = 0
            best_path = f'{CFG.MODEL_DIR}/best_model.pt'
            save_checkpoint({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_m,
                'config':      CFG.__dict__,
            }, best_path)
            print(f'  ★ Best! avg_f1={avg_f1:.4f}')
            # Sync best model lên Drive ngay lập tức
            sync_to_drive(CFG.MODEL_DIR,
                          f'/content/drive/MyDrive/tiki_absa/models/phobert')
        else:
            no_improve += 1
            if no_improve >= CFG.PATIENCE:
                print(f'\n  Early stopping (patience={CFG.PATIENCE}).')
                break

        # Sync checkpoint epoch này lên Drive
        sync_to_drive(CFG.CHECKPOINT_DIR,
                      f'/content/drive/MyDrive/tiki_absa/checkpoints/phobert')

    # 7. Test
    print('\n[5] Đánh giá trên TEST set (best model)...')
    best_model_path = f'{CFG.MODEL_DIR}/best_model.pt'
    # Fallback: nếu best_model bị xóa, dùng checkpoint epoch mới nhất
    if not os.path.exists(best_model_path):
        print('  [!] best_model.pt không tìm thấy, dùng checkpoint mới nhất...')
        ckpt_fallback, _ = load_latest_checkpoint(CFG.CHECKPOINT_DIR)
        if ckpt_fallback is None:
            print('  [!] Không có checkpoint nào. Bỏ qua test.')
            return
        model.load_state_dict(ckpt_fallback['model_state'])
        # Tái tạo best_model.pt từ checkpoint này
        save_checkpoint({'epoch': _, 'model_state': ckpt_fallback['model_state'],
                         'val_metrics': ckpt_fallback.get('val_metrics', {}),
                         'config': CFG.__dict__},
                        best_model_path)
        print(f'  [RECOVER] best_model.pt đã được tạo lại từ checkpoint epoch {_}')
    else:
        best_ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state'])
    test_m = evaluate(model, test_loader, device)

    print('\n' + '='*55)
    print('  KẾT QUẢ FINAL — PhoBERT (Test Set)')
    print('='*55)
    print(f'  AD Precision : {test_m["ad_precision"]:.4f}')
    print(f'  AD Recall    : {test_m["ad_recall"]:.4f}')
    print(f'  AD F1        : {test_m["ad_f1"]:.4f}')
    print(f'  AP Precision : {test_m["ap_precision"]:.4f}')
    print(f'  AP Recall    : {test_m["ap_recall"]:.4f}')
    print(f'  AP F1        : {test_m["ap_f1"]:.4f}')
    print(f'  Average F1   : {test_m["avg_f1"]:.4f}')
    print('='*55)

    # Lưu kết quả chi tiết
    results = {
        'model':    'PhoBERT',
        'best_val': best_metrics,
        'test':     test_m,
        'config': {
            'model_name':   CFG.PHOBERT_NAME,
            'last_n_layers': CFG.LAST_N_LAYERS,
            'batch_size':   CFG.BATCH_SIZE,
            'lr':           CFG.LR,
            'epochs':       CFG.EPOCHS,
            'focal_gamma':  CFG.FOCAL_GAMMA,
        },
    }
    os.makedirs('results', exist_ok=True)
    with open('results/phobert_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n  Kết quả: results/phobert_results.json')
    print('  Model  : models/phobert/best_model.pt')


if __name__ == '__main__':
    main()
