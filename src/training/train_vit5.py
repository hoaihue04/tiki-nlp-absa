"""
train_vit5_colab.py  (v5 - weighted loss + Colab + logging)
============================================================
"""
import os, sys, json, random, re, csv, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datetime import datetime
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ════════════════════════════════════════════════════════════════════════════
# 0. COLAB SETUP — tự động mount Drive + install packages
# ════════════════════════════════════════════════════════════════════════════
def setup_colab():
    """
    Phát hiện môi trường Colab, mount Drive, install packages thiếu.
    Trả về: (is_colab: bool, drive_root: str hoặc None)
    """
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")

    if not is_colab:
        print("ℹ️  Không chạy trên Colab → bỏ qua bước mount Drive")
        return False, None

    print("🔵 Google Colab detected!")

    # Mount Drive
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted tại /content/drive")
    except Exception as e:
        print(f"⚠️  Không mount được Drive: {e}")
        print("   Tiếp tục với /content (dữ liệu sẽ mất khi session kết thúc)")
        return True, None

    # Install packages nếu thiếu
    missing = []
    for pkg in ["sentencepiece"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"📦 Cài thêm packages: {missing}")
        os.system(f"pip install -q {' '.join(missing)}")

    drive_root = "/content/drive/MyDrive/TIKI"
    print(f"📁 Drive root: {drive_root}")
    return True, drive_root


IS_COLAB, DRIVE_ROOT = setup_colab()

# ─── Config ──────────────────────────────────────────────────────────────────
# Colab: dùng Drive nếu available, fallback về /content
if IS_COLAB and DRIVE_ROOT:
    _BASE       = DRIVE_ROOT
    DATA_DIR    = os.path.join(_BASE, "data/training")
    MODEL_DIR   = os.path.join(_BASE, "models/vit5")
    RESULTS_DIR = os.path.join(_BASE, "results")
else:
    DATA_DIR    = "data/training"
    MODEL_DIR   = "models/vit5"
    RESULTS_DIR = "tiki/results"

LABEL_MAP  = os.path.join(DATA_DIR, "label_map.json")
VIT5_MODEL = "VietAI/vit5-base"
EPOCHS     = 20
BATCH_SIZE = 8          # T4: 8 OK; A100: có thể tăng lên 16
LR         = 5e-4
MAX_SRC    = 128
MAX_TGT    = 128
SEED       = 42
PREFIX     = "vi2vi: "
RUN_ENV    = "colab"

SENTIMENT_STRINGS = {
    "positive": "tích cực",
    "negative": "tiêu cực",
    "neutral":  "trung lập",
}

_TF_VER = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


# ════════════════════════════════════════════════════════════════════════════
# 1. LOGGING — ghi ra console + file (Drive-safe)
# ════════════════════════════════════════════════════════════════════════════
def setup_logging(results_dir, run_ts, env):
    """Tạo logger + các file path. An toàn với Drive latency."""
    os.makedirs(results_dir, exist_ok=True)

    log_path     = os.path.join(results_dir, f"train_vit5_{env}_{run_ts}.log")
    metrics_path = os.path.join(results_dir, f"metrics_{env}_{run_ts}.json")
    epoch_csv    = os.path.join(results_dir, f"epoch_log_{env}_{run_ts}.csv")

    logger = logging.getLogger(f"vit5_{env}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)

    # Header CSV
    with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "eval_loss",
            "eval_f1", "eval_precision", "eval_recall",
            "learning_rate", "epoch_time_sec",
            "vram_allocated_gb", "vram_reserved_gb",
        ])

    logger.info(f"📝 Log file    : {log_path}")
    logger.info(f"📊 Metrics file: {metrics_path}")
    logger.info(f"📈 Epoch CSV   : {epoch_csv}")

    return logger, log_path, metrics_path, epoch_csv


# ════════════════════════════════════════════════════════════════════════════
# 2. EPOCH LOGGER CALLBACK (Colab: thêm VRAM tracking)
# ════════════════════════════════════════════════════════════════════════════
class EpochLoggerCallback(TrainerCallback):
    """Ghi thông tin từng epoch vào CSV. Colab version: thêm VRAM tracking."""

    def __init__(self, csv_path, logger, device):
        self.csv_path = csv_path
        self.logger   = logger
        self.device   = device
        self._t_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        import time
        self._t_start = time.time()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import time
        elapsed = time.time() - self._t_start if self._t_start else 0

        history    = state.log_history
        train_loss = next(
            (h.get("loss", "") for h in reversed(history)
             if "loss" in h and "eval_loss" not in h), ""
        )
        eval_loss = metrics.get("eval_loss", "")
        eval_f1   = metrics.get("eval_f1", "")
        eval_prec = metrics.get("eval_precision", "")
        eval_rec  = metrics.get("eval_recall", "")
        lr        = next(
            (h.get("learning_rate", "") for h in reversed(history)
             if "learning_rate" in h), ""
        )
        epoch = int(state.epoch) if state.epoch else ""

        # VRAM tracking
        vram_alloc = vram_resrv = ""
        if self.device == "cuda":
            vram_alloc = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            vram_resrv = round(torch.cuda.memory_reserved(0) / 1e9, 2)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, eval_loss,
                eval_f1, eval_prec, eval_rec,
                lr, f"{elapsed:.1f}",
                vram_alloc, vram_resrv,
            ])

        vram_str = f" | VRAM={vram_alloc}GB" if vram_alloc != "" else ""
        self.logger.info(
            f"📈 Epoch {epoch:>2} | "
            f"train_loss={train_loss if train_loss != '' else 'N/A':>7} | "
            f"eval_loss={eval_loss if eval_loss != '' else 'N/A':>7} | "
            f"F1={eval_f1 if eval_f1 != '' else 'N/A':>6} | "
            f"time={elapsed:.0f}s{vram_str}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. COLAB GPU DETECTION
# T4 (sm_75, 16GB): fp16 | L4 (sm_89, 22GB): bf16 | A100 (sm_80, 40GB): bf16
# ════════════════════════════════════════════════════════════════════════════
def detect_device_and_precision(logger):
    if not torch.cuda.is_available():
        logger.warning("⚠️  Không có GPU trên Colab!")
        logger.warning("   Vào: Runtime → Change runtime type → T4 GPU")
        return "cpu", False, False

    device     = "cuda"
    gpu_name   = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    vram_gb    = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"✅ Colab GPU   : {gpu_name}")
    logger.info(f"   Capability  : sm_{capability[0]}{capability[1]}")
    logger.info(f"   VRAM        : {vram_gb:.1f} GB")

    use_bf16 = capability[0] >= 8 and torch.cuda.is_bf16_supported()
    use_fp16 = (not use_bf16) and capability[0] >= 7

    tier = "A100/L4 (bf16)" if use_bf16 else \
           ("T4/V100 (fp16)" if use_fp16 else "unknown (fp32)")
    logger.info(f"   Detected as : {tier}")

    if vram_gb >= 35:
        logger.info("   💡 A100 detected! Có thể tăng BATCH_SIZE=16 để nhanh hơn")

    return device, use_fp16, use_bf16


# ════════════════════════════════════════════════════════════════════════════
# 4. LOAD TOKENIZER (cache về /content để tránh re-download)
# ════════════════════════════════════════════════════════════════════════════
def load_tokenizer(model_name, logger):
    cache_dir = "/content/hf_cache" if IS_COLAB else None

    attempts = [
        lambda: AutoTokenizer.from_pretrained(
            model_name, use_fast=False, legacy=False, cache_dir=cache_dir),
        lambda: AutoTokenizer.from_pretrained(
            model_name, use_fast=False, legacy=True, cache_dir=cache_dir),
        lambda: AutoTokenizer.from_pretrained(
            model_name, use_fast=False, cache_dir=cache_dir),
    ]
    for i, attempt in enumerate(attempts, 1):
        try:
            tok = attempt()
            logger.info(f"   ✅ Cách {i}: {type(tok).__name__} | vocab={tok.vocab_size:,}")
            return tok
        except Exception as e:
            logger.warning(f"   ⚠️  Cách {i} thất bại: {str(e)[:80]}")
    raise RuntimeError(
        "Không load được tokenizer!\n"
        "Thử: !pip install sentencepiece transformers==4.40.0"
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. SENTIMENT TOKEN IDs
# ════════════════════════════════════════════════════════════════════════════
def get_sentiment_token_ids(tokenizer, sents, sent_strings, logger):
    sent_token_ids = {}
    for sent in sents:
        sent_str = sent_strings.get(sent, sent)
        ids = tokenizer.encode(sent_str, add_special_tokens=False)
        sent_token_ids[sent] = ids
        logger.info(f"   '{sent_str}' → token ids: {ids}")
    return sent_token_ids


# ════════════════════════════════════════════════════════════════════════════
# 6. WEIGHTED LOSS
# ════════════════════════════════════════════════════════════════════════════
class WeightedSeq2SeqLoss(nn.Module):
    def __init__(self, sent_token_ids, sent_weights, vocab_size,
                 pad_token_id, label_smoothing=0.1):
        super().__init__()
        self.vocab_size      = vocab_size
        self.pad_token_id    = pad_token_id
        self.label_smoothing = label_smoothing

        token_weight_map = torch.ones(vocab_size)
        for sent_name, token_ids in sent_token_ids.items():
            w = sent_weights.get(sent_name, 1.0)
            for tid in token_ids:
                if 0 <= tid < vocab_size:
                    token_weight_map[tid] = max(token_weight_map[tid].item(), w)

        self.register_buffer("token_weight_map", token_weight_map)

    def forward(self, logits, labels):
        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        labels_flat = labels.reshape(-1)

        if self.label_smoothing > 0:
            loss_per_token = self._smooth_cross_entropy(logits_flat, labels_flat)
        else:
            loss_per_token = F.cross_entropy(
                logits_flat, labels_flat, ignore_index=-100, reduction="none"
            )

        valid_mask  = (labels_flat != -100).float()
        labels_safe = labels_flat.clone()
        labels_safe[labels_safe == -100] = 0

        weight_mask = self.token_weight_map[labels_safe] * valid_mask
        n_valid     = valid_mask.sum().clamp(min=1)
        return (loss_per_token * weight_mask).sum() / n_valid

    def _smooth_cross_entropy(self, logits, labels):
        eps       = self.label_smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss  = F.nll_loss(log_probs, labels.clamp(min=0),
                               ignore_index=-100, reduction="none")
        smooth    = -log_probs.mean(dim=-1)
        loss      = (1 - eps) * nll_loss + eps * smooth
        return loss * (labels != -100).float()


# ════════════════════════════════════════════════════════════════════════════
# 7. CUSTOM TRAINER
# ════════════════════════════════════════════════════════════════════════════
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, weighted_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted_loss_fn = weighted_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.weighted_loss_fn is None:
            return super().compute_loss(model, inputs,
                                        return_outputs=return_outputs, **kwargs)
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = self.weighted_loss_fn(outputs.logits, labels)
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


# ════════════════════════════════════════════════════════════════════════════
# 8. DATASET
# ════════════════════════════════════════════════════════════════════════════
class ViT5Dataset(Dataset):
    def __init__(self, df, tokenizer, max_src=MAX_SRC, max_tgt=MAX_TGT):
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt
        self.data      = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src = PREFIX + str(row["input_text"])
        tgt = str(row["target_text"])

        enc = self.tokenizer(src, max_length=self.max_src, truncation=True,
                             padding="max_length", return_tensors="pt")
        tgt_enc = self.tokenizer(tgt, max_length=self.max_tgt, truncation=True,
                                 padding="max_length", return_tensors="pt")
        labels = tgt_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ════════════════════════════════════════════════════════════════════════════
# 9. METRICS
# ════════════════════════════════════════════════════════════════════════════
def parse_quadruples(text):
    quads = set()
    for m in re.finditer(r"\(([^)]+)\)", text):
        parts = [p.strip().lower() for p in m.group(1).split(",")]
        if len(parts) == 4:
            quads.add(tuple(parts))
    return quads

def compute_f1_exact(preds, refs):
    tp = fp = fn = 0
    for p_str, r_str in zip(preds, refs):
        p_set = parse_quadruples(p_str)
        r_set = parse_quadruples(r_str)
        tp += len(p_set & r_set)
        fp += len(p_set - r_set)
        fn += len(r_set - p_set)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}

def decode_safe(tokenizer, ids_array):
    vocab_size  = getattr(tokenizer, "vocab_size", None) or 32128
    ids_clipped = np.clip(ids_array, 0, vocab_size - 1).astype(np.int32)
    return tokenizer.batch_decode(ids_clipped, skip_special_tokens=True)

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = decode_safe(tokenizer, preds)
        labels_fixed   = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_fixed, skip_special_tokens=True)
        m = compute_f1_exact(decoded_preds, decoded_labels)
        return {
            "eval_f1":        round(m["f1"],        4),
            "eval_precision": round(m["precision"], 4),
            "eval_recall":    round(m["recall"],    4),
        }
    return compute_metrics


# ════════════════════════════════════════════════════════════════════════════
# 10. TRAINING ARGS
# Colab: num_workers=2 (safe với Colab RAM limit)
# ════════════════════════════════════════════════════════════════════════════
def build_training_args(n_train_steps, device, use_fp16, use_bf16):
    warmup_steps = max(1, int(0.1 * n_train_steps))
    pad_multiple = 8 if (use_fp16 or use_bf16) else None
    num_workers  = 2 if device == "cuda" else 0   # Colab: 2 là safe

    kwargs = dict(
        output_dir                  = MODEL_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        warmup_steps                = warmup_steps,
        weight_decay                = 0.01,
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_f1",
        greater_is_better           = True,
        predict_with_generate       = True,
        generation_max_length       = MAX_TGT,
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        dataloader_num_workers      = num_workers,
        dataloader_pin_memory       = (device == "cuda"),
        gradient_checkpointing      = (device == "cuda"),
        logging_steps               = 50,
        save_total_limit            = 2,       # Drive space hạn chế
        seed                        = SEED,
        report_to                   = "none",  # tắt wandb/tensorboard
    )
    if _TF_VER >= (4, 41):
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**kwargs)

def build_trainer(model, args, train_ds, val_ds, tokenizer,
                  data_collator, compute_metrics_fn,
                  weighted_loss_fn, extra_callbacks):
    common = dict(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = data_collator,
        compute_metrics = compute_metrics_fn,
        callbacks       = [
            EarlyStoppingCallback(early_stopping_patience=3),
            *extra_callbacks,
        ],
        weighted_loss_fn = weighted_loss_fn,
    )
    if _TF_VER >= (4, 46):
        common["processing_class"] = tokenizer
    else:
        common["tokenizer"] = tokenizer
    return WeightedSeq2SeqTrainer(**common)


# ════════════════════════════════════════════════════════════════════════════
# 11. MAIN TRAIN
# ════════════════════════════════════════════════════════════════════════════
def train():
    # ── Logging ───────────────────────────────────────────────────────────────
    logger, log_path, metrics_path, epoch_csv = setup_logging(
        RESULTS_DIR, _RUN_TS, RUN_ENV
    )

    logger.info("=" * 65)
    logger.info(f"  ViT5 ASQP Training  —  ENV: {RUN_ENV.upper()}  —  {_RUN_TS}")
    logger.info("=" * 65)
    logger.info(f"🔧 transformers : {transformers.__version__}")
    logger.info(f"🔧 torch        : {torch.__version__}")
    logger.info(f"📦 Model        : {VIT5_MODEL}")
    logger.info(f"📁 Data dir     : {DATA_DIR}")
    logger.info(f"💾 Model dir    : {MODEL_DIR}")
    logger.info(f"📝 Results dir  : {RESULTS_DIR}")

    if IS_COLAB:
        logger.info(f"☁️  Colab Drive  : {'mounted → ' + str(DRIVE_ROOT) if DRIVE_ROOT else 'NOT mounted'}")

    # ── GPU ───────────────────────────────────────────────────────────────────
    device, use_fp16, use_bf16 = detect_device_and_precision(logger)
    precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
    logger.info(f"⚡ Precision    : {precision}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # ── Label map ─────────────────────────────────────────────────────────────
    with open(LABEL_MAP, encoding="utf-8") as f:
        lmap = json.load(f)

    sents       = lmap["sentiments"]
    sent_w_list = lmap.get("sent_weights_list")

    if sent_w_list:
        sent_weights_dict = {s: sent_w_list[i] for i, s in enumerate(sents)}
        logger.info("⚖️  Sentiment class weights:")
        for s, w in sent_weights_dict.items():
            logger.info(f"     {s:<12} → {w:.4f}")
    else:
        logger.warning("⚠️  Không có class weights → chạy lại prepare_data.py!")
        sent_weights_dict = {s: 1.0 for s in sents}

    # ── Load data ─────────────────────────────────────────────────────────────
    dfs = {}
    for split in ["train", "val", "test"]:
        path       = os.path.join(DATA_DIR, f"vit5_{split}.csv")
        dfs[split] = pd.read_csv(path)
        logger.info(f"📂 {split}: {len(dfs[split])} examples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"\n🔤 Load tokenizer ({VIT5_MODEL})...")
    tokenizer = load_tokenizer(VIT5_MODEL, logger)

    logger.info("\n🔍 Token ids của các từ sentiment:")
    sent_token_ids = get_sentiment_token_ids(tokenizer, sents,
                                             SENTIMENT_STRINGS, logger)

    # ── Weighted loss ─────────────────────────────────────────────────────────
    vocab_size    = getattr(tokenizer, "vocab_size", None) or 32128
    weighted_loss = WeightedSeq2SeqLoss(
        sent_token_ids  = sent_token_ids,
        sent_weights    = sent_weights_dict,
        vocab_size      = vocab_size,
        pad_token_id    = tokenizer.pad_token_id,
        label_smoothing = 0.1,
    ).to(device)
    logger.info(f"\n   WeightedSeq2SeqLoss OK | vocab={vocab_size:,} | ls=0.1")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"\n🔨 Load model {VIT5_MODEL}...")
    cache_dir = "/content/hf_cache" if IS_COLAB else None
    model     = AutoModelForSeq2SeqLM.from_pretrained(VIT5_MODEL, cache_dir=cache_dir)

    if device == "cuda":
        model.gradient_checkpointing_enable()
        logger.info("   ✅ Gradient checkpointing: BẬT")

    model.to(device)

    if device == "cuda":
        alloc = torch.cuda.memory_allocated(0) / 1e9
        resrv = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"   VRAM: {alloc:.2f}GB allocated / {resrv:.2f}GB reserved")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Tham số: {n_params:,} ({n_params/1e6:.0f}M)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ViT5Dataset(dfs["train"], tokenizer)
    val_ds   = ViT5Dataset(dfs["val"],   tokenizer)
    test_ds  = ViT5Dataset(dfs["test"],  tokenizer)

    pad_multiple  = 8 if (use_fp16 or use_bf16) else None
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=pad_multiple
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    n_train_steps   = (len(train_ds) // BATCH_SIZE) * EPOCHS
    training_args   = build_training_args(n_train_steps, device, use_fp16, use_bf16)
    epoch_logger_cb = EpochLoggerCallback(epoch_csv, logger, device)

    trainer = build_trainer(
        model              = model,
        args               = training_args,
        train_ds           = train_ds,
        val_ds             = val_ds,
        tokenizer          = tokenizer,
        data_collator      = data_collator,
        compute_metrics_fn = make_compute_metrics(tokenizer),
        weighted_loss_fn   = weighted_loss,
        extra_callbacks    = [epoch_logger_cb],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        f"\n🚀 Training ViT5 | epochs={EPOCHS} | batch={BATCH_SIZE} | "
        f"lr={LR} | precision={precision} | weighted_loss=YES"
    )
    if IS_COLAB:
        logger.info("   ⚠️  Tip: Đừng để màn hình tắt → Colab session bị disconnect!")

    train_start   = datetime.now()
    trainer.train()
    train_elapsed = (datetime.now() - train_start).total_seconds()
    logger.info(f"\n⏱️  Total training time: {train_elapsed/60:.1f} min")

    # ── Test ──────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
    logger.info("=" * 55)

    test_out     = trainer.predict(test_ds)
    preds        = decode_safe(tokenizer, test_out.predictions)
    labels_fixed = np.where(
        test_out.label_ids != -100,
        test_out.label_ids,
        tokenizer.pad_token_id,
    )
    refs    = tokenizer.batch_decode(labels_fixed, skip_special_tokens=True)
    metrics = compute_f1_exact(preds, refs)

    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")

    logger.info("\n🔍 Ví dụ dự đoán (5 mẫu đầu):")
    logger.info("-" * 70)
    for i in range(min(5, len(preds))):
        logger.info(f"  Input : {dfs['test']['input_text'].iloc[i]}")
        logger.info(f"  Target: {refs[i]}")
        logger.info(f"  Pred  : {preds[i]}")
        ok = parse_quadruples(preds[i]) == parse_quadruples(refs[i])
        logger.info(f"  Match : {'✅' if ok else '❌'}")
        logger.info("")

    # ── Lưu metrics JSON ─────────────────────────────────────────────────────
    gpu_info = {}
    if device == "cuda":
        gpu_info = {
            "gpu_name":   torch.cuda.get_device_name(0),
            "vram_gb":    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "capability": list(torch.cuda.get_device_capability(0)),
        }

    result_payload = {
        "run_timestamp":      _RUN_TS,
        "environment":        RUN_ENV,
        "model":              VIT5_MODEL,
        "precision":          precision,
        "epochs":             EPOCHS,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LR,
        "training_time_min":  round(train_elapsed / 60, 2),
        "test_precision":     round(float(metrics["precision"]), 4),
        "test_recall":        round(float(metrics["recall"]),    4),
        "test_f1":            round(float(metrics["f1"]),        4),
        "used_class_weights": sent_w_list is not None,
        "sent_weights_used":  sent_w_list,
        "gpu_info":           gpu_info,
        "colab_drive_root":   DRIVE_ROOT,
        "log_file":           log_path,
        "epoch_csv":          epoch_csv,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"\n💾 Metrics saved → {metrics_path}")

    # Cũng lưu vào model dir để tương thích với code cũ
    with open(os.path.join(MODEL_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    logger.info(f"✅ Xong! Model    → {MODEL_DIR}/")
    logger.info(f"📝 Log đầy đủ    → {log_path}")
    logger.info(f"📈 Epoch log CSV → {epoch_csv}")

    if IS_COLAB and DRIVE_ROOT:
        logger.info(f"☁️  Tất cả đã sync lên Google Drive: {RESULTS_DIR}")


if __name__ == "__main__":
    train()