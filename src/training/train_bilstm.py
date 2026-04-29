#!/usr/bin/env python3

import json
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    print('[CẢNH BÁO] torchcrf chưa cài → pip install torchcrf (CRF head tắt)')
    HAS_CRF = False

# AMP (Mixed Precision)
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Config ────────────────────────────────────────────────────────────────
class Config:
    DATA_DIR       = 'data/training'
    CHECKPOINT_DIR = 'checkpoints/bilstm'
    MODEL_DIR      = 'models/bilstm'

    # [OPT] Model nhỏ hơn — nhanh hơn ~4x, F1 gần tương đương
    EMBED_DIM      = 128   # 256 → 128
    HIDDEN_DIM     = 128   # 256 → 128  (LSTM cost ∝ hidden²)
    NUM_LAYERS     = 2
    DROPOUT        = 0.3
    NUM_HEADS      = 2     # 4 → 2
    MAX_LEN        = 100   # 150 → 100  (câu VN thường < 60 từ)

    # [OPT] Training nhanh hơn
    BATCH_SIZE     = 128   # 64 → 128
    EPOCHS         = 25    # 40 → 25
    LR             = 1e-3
    WEIGHT_DECAY   = 1e-4
    GRAD_CLIP      = 5.0
    WARMUP_STEPS   = 100
    PATIENCE       = 6     # 8 → 6
    SAVE_EVERY     = 2     # lưu checkpoint mỗi 2 epoch
    EVAL_EVERY     = 2     # [OPT] evaluate validation mỗi 2 epoch

    # Loss
    FOCAL_GAMMA    = 2.0
    CRF_WEIGHT     = 0.2
    CLS_WEIGHT     = 1.0

    NUM_CATS       = 17
    NUM_SENT       = 4     # none/pos/neu/neg
    NUM_BIO        = 7


CFG = Config()

BIO_NAMES  = ['O', 'B-positive', 'I-positive',
              'B-neutral', 'I-neutral', 'B-negative', 'I-negative']
BIO2IDX    = {t: i for i, t in enumerate(BIO_NAMES)}


# ══════════════════════════════════════════════════════════════════════════
# DATASET — Pre-compute IDs khi load (không tính lại trong __getitem__)
# ══════════════════════════════════════════════════════════════════════════
class BiLSTMDataset(Dataset):
    def __init__(self, file_path: str, vocab: dict, max_len: int = CFG.MAX_LEN):
        self.samples = []
        unk_id = vocab.get('<UNK>', 1)
        self._load(file_path, vocab, unk_id, max_len)

    def _load(self, path: str, vocab: dict, unk_id: int, max_len: int):
        toks_buf, bio_buf, labels_buf = [], [], None

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('# labels:'):
                    labels_buf = list(map(int, line.split(': ', 1)[1].split()))
                elif line == '':
                    if toks_buf and labels_buf is not None:
                        # [OPT] Pre-compute IDs ngay tại đây
                        t = toks_buf[:max_len]
                        b = bio_buf[:max_len]
                        self.samples.append((
                            torch.tensor([vocab.get(w, unk_id) for w in t], dtype=torch.long),
                            torch.tensor([BIO2IDX.get(tag, 0) for tag in b],  dtype=torch.long),
                            torch.tensor(labels_buf, dtype=torch.long),
                            len(t),
                        ))
                    toks_buf, bio_buf, labels_buf = [], [], None
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        toks_buf.append(parts[0])
                        bio_buf.append(parts[1])

        if toks_buf and labels_buf is not None:
            t = toks_buf[:max_len]
            b = bio_buf[:max_len]
            self.samples.append((
                torch.tensor([vocab.get(w, unk_id) for w in t], dtype=torch.long),
                torch.tensor([BIO2IDX.get(tag, 0) for tag in b],  dtype=torch.long),
                torch.tensor(labels_buf, dtype=torch.long),
                len(t),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (token_ids, bio_ids, labels, length) — đã pre-computed


def collate_fn(batch):
    max_len   = max(b[3] for b in batch)
    B         = len(batch)
    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    bio_ids   = torch.zeros(B, max_len, dtype=torch.long)
    mask      = torch.zeros(B, max_len, dtype=torch.bool)
    labels    = torch.stack([b[2] for b in batch])

    for i, (tids, bids, _, L) in enumerate(batch):
        token_ids[i, :L] = tids
        bio_ids[i, :L]   = bids
        mask[i, :L]      = True

    return token_ids, bio_ids, mask, labels


# ══════════════════════════════════════════════════════════════════════════
# MODEL — Batched classifiers + nn.MultiheadAttention
# ══════════════════════════════════════════════════════════════════════════
class BiLSTMCRFModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        H = cfg.HIDDEN_DIM * 2  # BiLSTM bidirectional output

        self.embedding   = nn.Embedding(vocab_size, cfg.EMBED_DIM, padding_idx=0)
        self.emb_dropout = nn.Dropout(cfg.DROPOUT)

        self.bilstm = nn.LSTM(
            input_size=cfg.EMBED_DIM,
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.DROPOUT if cfg.NUM_LAYERS > 1 else 0.0,
        )
        self.layer_norm   = nn.LayerNorm(H)
        self.lstm_dropout = nn.Dropout(cfg.DROPOUT)

        # [OPT] Dùng PyTorch built-in MHA thay vì custom implementation
        self.attention = nn.MultiheadAttention(
            embed_dim=H, num_heads=cfg.NUM_HEADS,
            dropout=cfg.DROPOUT * 0.5, batch_first=True
        )

        # CRF auxiliary
        if HAS_CRF:
            self.emission_proj = nn.Linear(H, cfg.NUM_BIO)
            self.crf           = CRF(cfg.NUM_BIO, batch_first=True)
        else:
            self.emission_proj = None
            self.crf           = None

        # [OPT] Batched classifier: 1 Linear thay vì 17 Sequential riêng
        # Forward: Linear(H, 17*4) → view(B, 17, 4)
        self.cls_dropout = nn.Dropout(cfg.DROPOUT)
        self.classifier  = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT * 0.5),
            nn.Linear(H // 2, cfg.NUM_CATS * cfg.NUM_SENT),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor):
        """
        token_ids: [B, S]
        mask     : [B, S] bool
        → logits : [B, 17, 4]
        → emission: [B, S, 7] or None
        """
        emb = self.emb_dropout(self.embedding(token_ids))  # [B, S, E]

        lengths = mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        out, _  = self.bilstm(packed)
        out, _  = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=token_ids.size(1))

        out = self.layer_norm(out)
        out = self.lstm_dropout(out)              # [B, S, H]

        # [OPT] PyTorch MHA — key_padding_mask dùng bool ~mask
        key_pad = ~mask                            # True = padding position
        ctx, _  = self.attention(out, out, out, key_padding_mask=key_pad)
        # Mean pooling chỉ trên valid tokens
        valid_len = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        sent_repr = (ctx * mask.unsqueeze(-1).float()).sum(dim=1) / valid_len  # [B, H]

        # CRF emission
        emission = self.emission_proj(out) if self.emission_proj is not None else None

        # [OPT] Batched classification
        h      = self.cls_dropout(sent_repr)
        logits = self.classifier(h)               # [B, 17*4]
        logits = logits.view(-1, CFG.NUM_CATS, CFG.NUM_SENT)  # [B, 17, 4]

        return logits, emission


# ══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS — Batched trên tất cả 17 heads cùng lúc
# ══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, 17, 4]
        targets: [B, 17]
        """
        B, C, S = logits.shape
        # Flatten để tính CE: [B*17, 4] vs [B*17]
        ce   = F.cross_entropy(logits.view(-1, S), targets.view(-1), reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma * ce).mean()
        return loss


focal_loss_fn = FocalLoss(gamma=CFG.FOCAL_GAMMA)


# ══════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []

    for token_ids, _, mask, labels in loader:
        token_ids = token_ids.to(device)
        mask      = mask.to(device)

        logits, _ = model(token_ids, mask)         # [B, 17, 4]
        preds     = logits.argmax(dim=-1).cpu().numpy()  # [B, 17]
        all_true.append(labels.numpy())
        all_pred.append(preds)

    y_true = np.concatenate(all_true)   # [N, 17]
    y_pred = np.concatenate(all_pred)

    # AD: binary detection
    y_true_ad = (y_true != 0).astype(int).flatten()
    y_pred_ad = (y_pred != 0).astype(int).flatten()
    ad_p  = precision_score(y_true_ad, y_pred_ad, zero_division=0)
    ad_r  = recall_score(y_true_ad, y_pred_ad, zero_division=0)
    ad_f1 = f1_score(y_true_ad, y_pred_ad, zero_division=0)

    # AP: sentiment cho aspect thực sự xuất hiện
    mask_ap   = (y_true != 0)
    y_true_ap = y_true[mask_ap]
    y_pred_ap = y_pred[mask_ap]
    if len(y_true_ap) > 0:
        ap_p  = precision_score(y_true_ap, y_pred_ap, labels=[1,2,3],
                                average='micro', zero_division=0)
        ap_r  = recall_score(y_true_ap, y_pred_ap, labels=[1,2,3],
                             average='micro', zero_division=0)
        ap_f1 = f1_score(y_true_ap, y_pred_ap, labels=[1,2,3],
                         average='micro', zero_division=0)
    else:
        ap_p = ap_r = ap_f1 = 0.0

    return {
        'ad_precision': float(ad_p),
        'ad_recall':    float(ad_r),
        'ad_f1':        float(ad_f1),
        'ap_precision': float(ap_p),
        'ap_recall':    float(ap_r),
        'ap_f1':        float(ap_f1),
        'avg_f1':       float((ad_f1 + ap_f1) / 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None, 0
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith('epoch_') and f.endswith('.pt')]
    if not files:
        return None, 0
    epochs = sorted([int(f.replace('epoch_', '').replace('.pt', ''))
                     for f in files], reverse=True)
    for ep in epochs:
        path = os.path.join(ckpt_dir, f'epoch_{ep}.pt')
        try:
            state = torch.load(path, map_location='cpu')
            print(f'  [RESUME] Checkpoint epoch {ep}: {path}')
            return state, ep
        except Exception as e:
            print(f'  [!] epoch_{ep}.pt bị corrupt ({type(e).__name__}), bỏ qua...')
            os.remove(path)
    print('  [!] Tất cả checkpoint đều corrupt. Train từ đầu.')
    return None, 0


def _cleanup_checkpoints(ckpt_dir: str, keep_last: int = 3):
    """Giữ lại keep_last checkpoint mới nhất, xóa các epoch cũ hơn."""
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith('epoch_') and f.endswith('.pt')]
    if len(files) <= keep_last:
        return
    epochs = sorted([int(f.replace('epoch_', '').replace('.pt', ''))
                     for f in files])
    for ep in epochs[:-keep_last]:
        p = os.path.join(ckpt_dir, f'epoch_{ep}.pt')
        if os.path.exists(p):
            os.remove(p)


def sync_to_drive(local_dir: str, drive_dir: str):
    """Sync local dir lên Google Drive (chỉ chạy trong Colab)."""
    import shutil
    try:
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
# TRAIN EPOCH — AMP + batched loss
# ══════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    n = 0
    use_amp = HAS_AMP and device.type == 'cuda'

    for token_ids, bio_ids, mask, labels in loader:
        token_ids = token_ids.to(device, non_blocking=True)
        bio_ids   = bio_ids.to(device, non_blocking=True)
        mask      = mask.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        with (autocast() if use_amp else torch.no_grad.__class__()):  # type: ignore
            if not use_amp:
                # Plain forward khi không có AMP
                logits, emission = model(token_ids, mask)
                cls_loss = focal_loss_fn(logits, labels)

                crf_loss = torch.tensor(0.0, device=device)
                if emission is not None and HAS_CRF and model.crf is not None:
                    seq_len  = emission.size(1)
                    crf_ll   = model.crf(emission, bio_ids[:, :seq_len],
                                         mask=mask[:, :seq_len], reduction='mean')
                    crf_loss = -crf_ll

                loss = cfg.CLS_WEIGHT * cls_loss + cfg.CRF_WEIGHT * crf_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()
            else:
                with autocast():
                    logits, emission = model(token_ids, mask)
                    cls_loss = focal_loss_fn(logits, labels)

                    crf_loss = torch.tensor(0.0, device=device)
                    if emission is not None and HAS_CRF and model.crf is not None:
                        seq_len  = emission.size(1)
                        crf_ll   = model.crf(emission, bio_ids[:, :seq_len],
                                             mask=mask[:, :seq_len], reduction='mean')
                        crf_loss = -crf_ll

                    loss = cfg.CLS_WEIGHT * cls_loss + cfg.CRF_WEIGHT * crf_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = HAS_AMP and device.type == 'cuda'
    scaler  = GradScaler() if use_amp else None

    print(f'\n[BiLSTM] Device: {device}')
    if device.type == 'cuda':
        print(f'         GPU   : {torch.cuda.get_device_name(0)}')
    print(f'         AMP   : {"ON" if use_amp else "OFF"}')

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.MODEL_DIR, exist_ok=True)

    # 1. Vocab
    print('\n[1] Đọc vocab...')
    with open(f'{CFG.DATA_DIR}/vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f'    Vocab size: {len(vocab):,}')

    # 2. Dataset
    print('\n[2] Tải dataset (pre-computing IDs)...')
    train_ds = BiLSTMDataset(f'{CFG.DATA_DIR}/bilstm_train.txt', vocab)
    val_ds   = BiLSTMDataset(f'{CFG.DATA_DIR}/bilstm_val.txt',   vocab)
    test_ds  = BiLSTMDataset(f'{CFG.DATA_DIR}/bilstm_test.txt',  vocab)
    print(f'    Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')

    # [OPT] num_workers=2 cho Colab (0 trên Windows để tránh lỗi spawn)
    nw = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=nw, pin_memory=(device.type=='cuda'),
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE * 2,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CFG.BATCH_SIZE * 2,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Model
    print('\n[3] Khởi tạo model...')
    model      = BiLSTMCRFModel(len(vocab), CFG).to(device)
    total_p    = sum(p.numel() for p in model.parameters())
    print(f'    Parameters: {total_p:,}  '
          f'(HIDDEN={CFG.HIDDEN_DIM}, EMBED={CFG.EMBED_DIM}, MAX_LEN={CFG.MAX_LEN})')

    # 4. Optimizer + Scheduler
    optimizer    = Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    total_steps  = len(train_loader) * CFG.EPOCHS
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=CFG.WARMUP_STEPS)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=total_steps - CFG.WARMUP_STEPS,
                                     eta_min=1e-5)
    scheduler    = SequentialLR(optimizer,
                                schedulers=[warmup_sched, cosine_sched],
                                milestones=[CFG.WARMUP_STEPS])

    # 5. Resume
    start_epoch  = 1
    best_avg_f1  = 0.0
    no_improve   = 0
    best_metrics = {}
    last_val_m   = {}

    ckpt, last_ep = load_latest_checkpoint(CFG.CHECKPOINT_DIR)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state'])
        # Khong load optimizer/scheduler state — incompatible giua cac PyTorch versions
        start_epoch  = last_ep + 1
        best_avg_f1  = ckpt.get('best_avg_f1', 0.0)
        no_improve   = ckpt.get('no_improve', 0)
        best_metrics = ckpt.get('best_metrics', {})
        print(f'  Resumed epoch {last_ep}, best_avg_f1={best_avg_f1:.4f}')

    # 6. Training
    print(f'\n[4] Training từ epoch {start_epoch}/{CFG.EPOCHS}  '
          f'(evaluate mỗi {CFG.EVAL_EVERY} epoch)')
    print('-' * 65)

    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 scaler, device, CFG)
        elapsed    = time.time() - t0

        # [OPT] Chỉ evaluate mỗi EVAL_EVERY epoch
        do_eval = (epoch % CFG.EVAL_EVERY == 0) or (epoch == CFG.EPOCHS)
        if do_eval:
            val_m      = evaluate(model, val_loader, device)
            last_val_m = val_m
            avg_f1     = val_m['avg_f1']
            print(f'Epoch {epoch:02d}/{CFG.EPOCHS} | loss={train_loss:.4f} | '
                  f'AD={val_m["ad_f1"]:.4f} | AP={val_m["ap_f1"]:.4f} | '
                  f'avg={avg_f1:.4f} | t={elapsed:.1f}s ✓eval')
        else:
            avg_f1 = last_val_m.get('avg_f1', 0.0)
            print(f'Epoch {epoch:02d}/{CFG.EPOCHS} | loss={train_loss:.4f} | '
                  f't={elapsed:.1f}s')

        # Checkpoint mỗi SAVE_EVERY epoch
        if epoch % CFG.SAVE_EVERY == 0:
            ckpt_data = {
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_avg_f1':     best_avg_f1,
                'no_improve':      no_improve,
                'best_metrics':    best_metrics,
                'val_metrics':     last_val_m,
            }
            if scaler:
                ckpt_data['scaler_state'] = scaler.state_dict()
            save_checkpoint(ckpt_data, f'{CFG.CHECKPOINT_DIR}/epoch_{epoch}.pt')
            _cleanup_checkpoints(CFG.CHECKPOINT_DIR, keep_last=3)
            sync_to_drive(CFG.CHECKPOINT_DIR,
                          '/content/drive/MyDrive/tiki_absa/checkpoints/bilstm')

        # Best model + early stopping (chỉ khi có val metrics mới)
        if do_eval:
            if avg_f1 > best_avg_f1:
                best_avg_f1  = avg_f1
                best_metrics = last_val_m.copy()
                no_improve   = 0
                best_path = f'{CFG.MODEL_DIR}/best_model.pt'
                save_checkpoint({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'val_metrics': last_val_m,
                    'vocab_size':  len(vocab),
                    'config':      CFG.__dict__,
                }, best_path)
                print(f'  ★ Best! avg_f1={avg_f1:.4f}')
                sync_to_drive(CFG.MODEL_DIR,
                              '/content/drive/MyDrive/tiki_absa/models/bilstm')
            else:
                no_improve += 1
                # Tính đổi sang số epoch thực (mỗi lần check = EVAL_EVERY epoch)
                if no_improve * CFG.EVAL_EVERY >= CFG.PATIENCE * CFG.EVAL_EVERY:
                    print(f'\n  Early stopping (no improve {no_improve} lần eval).')
                    break

    # 7. Test
    print('\n[5] Test set evaluation...')
    best_model_path = f'{CFG.MODEL_DIR}/best_model.pt'
    if not os.path.exists(best_model_path):
        print('  [!] best_model.pt không tìm thấy, dùng checkpoint mới nhất...')
        ckpt_fallback, fallback_ep = load_latest_checkpoint(CFG.CHECKPOINT_DIR)
        if ckpt_fallback is None:
            print('  [!] Không có checkpoint nào. Bỏ qua test.')
            return
        model.load_state_dict(ckpt_fallback['model_state'])
        save_checkpoint({
            'epoch':       fallback_ep,
            'model_state': ckpt_fallback['model_state'],
            'val_metrics': ckpt_fallback.get('val_metrics', {}),
            'vocab_size':  len(vocab),
            'config':      CFG.__dict__,
        }, best_model_path)
        print(f'  [RECOVER] best_model.pt tạo lại từ epoch {fallback_ep}')
    else:
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    test_m = evaluate(model, test_loader, device)

    print('\n' + '='*55)
    print('  KẾT QUẢ FINAL — BiLSTM-CRF (Test Set)')
    print('='*55)
    print(f'  AD Precision : {test_m["ad_precision"]:.4f}')
    print(f'  AD Recall    : {test_m["ad_recall"]:.4f}')
    print(f'  AD F1        : {test_m["ad_f1"]:.4f}')
    print(f'  AP Precision : {test_m["ap_precision"]:.4f}')
    print(f'  AP Recall    : {test_m["ap_recall"]:.4f}')
    print(f'  AP F1        : {test_m["ap_f1"]:.4f}')
    print(f'  Average F1   : {test_m["avg_f1"]:.4f}')
    print('='*55)

    os.makedirs('results', exist_ok=True)
    with open('results/bilstm_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'model':    'BiLSTM-CRF',
            'best_val': best_metrics,
            'test':     test_m,
            'config': {
                'embed_dim':   CFG.EMBED_DIM,
                'hidden_dim':  CFG.HIDDEN_DIM,
                'max_len':     CFG.MAX_LEN,
                'batch_size':  CFG.BATCH_SIZE,
                'epochs':      CFG.EPOCHS,
                'focal_gamma': CFG.FOCAL_GAMMA,
                'amp':         use_amp,
            },
        }, f, ensure_ascii=False, indent=2)
    print('  Kết quả: results/bilstm_results.json')
    print('  Model  : models/bilstm/best_model.pt')


if __name__ == '__main__':
    main()
