#!/usr/bin/env python3
"""
train_svm_tfidf.py — SVM + TF-IDF cho ABSA (AD + AP)
======================================================
Mô hình baseline sử dụng TF-IDF vectorization + SVM (LinearSVC) cho:
  - AD (Aspect Detection): binary classifier cho mỗi trong 17 category
  - AP (Aspect Polarity) : 3-class sentiment classifier (pos/neu/neg)
                           chỉ trên các sample có aspect đó xuất hiện

Ưu điểm so với Deep Learning:
  - Không cần GPU
  - Train siêu nhanh (< 1 phút trên Colab CPU)
  - Tốt làm baseline để so sánh

Pipeline cho mỗi category i (0..16):
  ┌───────────────────────────────────────────────────────────┐
  │  Text → TF-IDF features                                   │
  │  → AD_clf[i]:  LinearSVC binary (0=none, 1=có khía cạnh) │
  │  → AP_clf[i]:  LinearSVC 3-class (1=pos, 2=neu, 3=neg)   │
  │              (chỉ train trên samples có label != 0)       │
  └───────────────────────────────────────────────────────────┘

Output:
    models/svm/svm_model.pkl       — model đã train
    results/svm_results.json       — metrics trên test set
"""

import json
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
class Config:
    DATA_DIR    = 'data/training'
    MODEL_DIR   = 'models/svm'
    RESULTS_DIR = 'results'

    # TF-IDF
    TFIDF_MAX_FEATURES = 50_000   # tối đa 50k từ
    TFIDF_NGRAM_RANGE  = (1, 3)   # unigram + bigram + trigram
    TFIDF_MIN_DF       = 2        # bỏ từ xuất hiện < 2 lần
    TFIDF_SUBLINEAR_TF = True     # log(1 + tf) thay vì tf

    # SVM
    SVM_C          = 1.0          # regularization strength
    SVM_MAX_ITER   = 2000
    SVM_CLASS_WEIGHT = 'balanced' # xử lý mất cân bằng nhãn

    # Drive sync path (Colab)
    DRIVE_MODEL_DIR = '/content/drive/MyDrive/tiki_absa/models/svm'


CFG = Config()

# 17 categories — thứ tự cố định (khớp với BiLSTM và PhoBERT)
CATEGORIES = [
    'PRODUCT#QUALITY',       'DELIVERY#SPEED',        'DELIVERY#PACKAGING',
    'PRICE#AFFORDABILITY',   'SELLER#SERVICE',         'PRODUCT#FUNCTION',
    'PRODUCT#COMFORT',       'PRODUCT#DESIGN',         'DELIVERY#ACCURACY',
    'PRODUCT#DURABILITY',    'PRODUCT#SAFETY',         'SELLER#AUTHENTICITY',
    'PRODUCT#MATERIAL',      'PRODUCT#SIZE',           'PRODUCT#VALUE',
    'PRICE#DISCOUNT',        'SELLER#RESPONSIVENESS',
]
NUM_CATS   = len(CATEGORIES)
SENT_NAMES = ['none', 'positive', 'neutral', 'negative']


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING — đọc từ phobert_*.csv (text + 17 label columns)
# ══════════════════════════════════════════════════════════════════════════
def load_csv_split(data_dir: str, split: str):
    """
    Đọc phobert_{split}.csv → (texts: List[str], labels: np.ndarray [N, 17])
    label values: 0=none, 1=positive, 2=neutral, 3=negative
    """
    path = os.path.join(data_dir, f'phobert_{split}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Không tìm thấy: {path}')

    df         = pd.read_csv(path, dtype=str)
    texts      = df['text'].fillna('').tolist()
    label_cols = [f'label_{i}' for i in range(NUM_CATS)]
    labels     = df[label_cols].fillna(0).astype(int).values  # [N, 17]

    return texts, labels


# ══════════════════════════════════════════════════════════════════════════
# EVALUATE FUNCTION — dùng chung logic với BiLSTM/PhoBERT
# ══════════════════════════════════════════════════════════════════════════
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính AD và AP metrics từ prediction arrays.
    y_true, y_pred: [N, 17], values 0..3 (0=none)
    """
    # ── AD: binary detection ──
    y_true_ad = (y_true != 0).astype(int).flatten()
    y_pred_ad = (y_pred != 0).astype(int).flatten()
    ad_p  = precision_score(y_true_ad, y_pred_ad, zero_division=0)
    ad_r  = recall_score(y_true_ad,    y_pred_ad, zero_division=0)
    ad_f1 = f1_score(y_true_ad,        y_pred_ad, zero_division=0)

    # ── AP: sentiment cho aspect thực sự xuất hiện ──
    mask_ap   = (y_true != 0)
    y_true_ap = y_true[mask_ap]
    y_pred_ap = y_pred[mask_ap]

    if len(y_true_ap) > 0:
        ap_p  = precision_score(y_true_ap, y_pred_ap, labels=[1, 2, 3],
                                average='micro', zero_division=0)
        ap_r  = recall_score(y_true_ap, y_pred_ap, labels=[1, 2, 3],
                             average='micro', zero_division=0)
        ap_f1 = f1_score(y_true_ap, y_pred_ap, labels=[1, 2, 3],
                         average='micro', zero_division=0)
    else:
        ap_p = ap_r = ap_f1 = 0.0

    # ── Per-category AD ──
    per_category = {}
    for i, cat in enumerate(CATEGORIES):
        yt = (y_true[:, i] != 0).astype(int)
        yp = (y_pred[:, i] != 0).astype(int)
        per_category[cat] = {
            'precision': float(precision_score(yt, yp, zero_division=0)),
            'recall':    float(recall_score(yt, yp, zero_division=0)),
            'f1':        float(f1_score(yt, yp, zero_division=0)),
            'support':   int(yt.sum()),
        }

    return {
        'ad_precision':  float(ad_p),
        'ad_recall':     float(ad_r),
        'ad_f1':         float(ad_f1),
        'ap_precision':  float(ap_p),
        'ap_recall':     float(ap_r),
        'ap_f1':         float(ap_f1),
        'avg_f1':        float((ad_f1 + ap_f1) / 2),
        'per_category':  per_category,
        'n_samples':     int(len(y_true)),
    }


# ══════════════════════════════════════════════════════════════════════════
# SVM MODEL CLASS
# ══════════════════════════════════════════════════════════════════════════
class SVMTFIDFModel:
    """
    Pipeline SVM + TF-IDF cho ABSA với 17 categories.
    Mỗi category có 2 classifier:
      - ad_clf[i]: binary (0 vs 1) — phát hiện aspect
      - ap_clf[i]: multiclass (1/2/3) — phân loại sentiment
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Một TF-IDF vectorizer dùng chung cho tất cả classifiers
        self.vectorizer = TfidfVectorizer(
            max_features  = cfg.TFIDF_MAX_FEATURES,
            ngram_range   = cfg.TFIDF_NGRAM_RANGE,
            min_df        = cfg.TFIDF_MIN_DF,
            sublinear_tf  = cfg.TFIDF_SUBLINEAR_TF,
            analyzer      = 'word',
            token_pattern = r'\S+',  # token bằng bất kỳ non-whitespace (hỗ trợ tiếng Việt đã tokenize)
        )

        # 17 AD classifiers (binary)
        self.ad_clfs = [None] * NUM_CATS
        # 17 AP classifiers (3-class: pos/neu/neg)
        self.ap_clfs = [None] * NUM_CATS

        self.is_fitted = False

    def fit(self, texts, labels):
        """
        texts : List[str], N samples
        labels: np.ndarray [N, 17], values 0..3
        """
        print('\n[TF-IDF] Đang fit vectorizer...')
        t0    = time.time()
        X     = self.vectorizer.fit_transform(texts)
        print(f'  Vocab size: {len(self.vectorizer.vocabulary_):,}  '
              f'| Features: {X.shape[1]:,}  | t={time.time()-t0:.1f}s')

        for i, cat in enumerate(CATEGORIES):
            # ── AD classifier ──
            y_ad = (labels[:, i] != 0).astype(int)
            n_pos = y_ad.sum()
            n_neg = len(y_ad) - n_pos

            if n_pos < 5:
                # Quá ít positive — dùng dummy classifier (luôn predict 0)
                print(f'  [AD-{i:02d}] {cat:<28} : skip (chỉ {n_pos} positive samples)')
                self.ad_clfs[i] = None
            else:
                clf = LinearSVC(
                    C            = self.cfg.SVM_C,
                    max_iter     = self.cfg.SVM_MAX_ITER,
                    class_weight = self.cfg.SVM_CLASS_WEIGHT,
                )
                clf.fit(X, y_ad)
                self.ad_clfs[i] = clf

            # ── AP classifier: chỉ train trên samples có aspect ──
            mask_ap = (labels[:, i] != 0)
            y_ap    = labels[mask_ap, i]  # values: 1, 2, 3

            if mask_ap.sum() < 10:
                # Quá ít samples cho AP
                self.ap_clfs[i] = None
                print(f'  [AP-{i:02d}] {cat:<28} : skip (chỉ {mask_ap.sum()} AP samples)')
                continue

            unique_classes = np.unique(y_ap)
            if len(unique_classes) < 2:
                # Chỉ có 1 class — ghi nhớ class đó
                self.ap_clfs[i] = int(unique_classes[0])  # store as constant
            else:
                X_ap  = X[mask_ap]
                clf_ap = LinearSVC(
                    C            = self.cfg.SVM_C,
                    max_iter     = self.cfg.SVM_MAX_ITER,
                    class_weight = self.cfg.SVM_CLASS_WEIGHT,
                )
                clf_ap.fit(X_ap, y_ap)
                self.ap_clfs[i] = clf_ap

        self.is_fitted = True
        print(f'\n  AD classifiers trained: {sum(c is not None for c in self.ad_clfs)}/17')
        print(f'  AP classifiers trained: {sum(isinstance(c, LinearSVC) for c in self.ap_clfs)}/17')

    def predict(self, texts) -> np.ndarray:
        """
        texts: List[str], N samples
        returns: np.ndarray [N, 17], values 0..3
        """
        if not self.is_fitted:
            raise RuntimeError('Model chưa được train. Gọi fit() trước.')

        X       = self.vectorizer.transform(texts)
        N       = X.shape[0]
        y_pred  = np.zeros((N, NUM_CATS), dtype=int)

        for i in range(NUM_CATS):
            # ── AD prediction ──
            if self.ad_clfs[i] is None:
                ad_pred = np.zeros(N, dtype=int)
            else:
                ad_pred = self.ad_clfs[i].predict(X)  # 0 or 1

            # ── AP prediction: chỉ predict cho samples được AD detect ──
            aspect_mask = (ad_pred == 1)
            if aspect_mask.sum() == 0:
                # Không detect aspect nào
                y_pred[:, i] = 0
                continue

            if self.ap_clfs[i] is None:
                # Không có AP clf → default positive (1)
                ap_for_detected = np.ones(aspect_mask.sum(), dtype=int)
            elif isinstance(self.ap_clfs[i], int):
                # Constant classifier
                ap_for_detected = np.full(aspect_mask.sum(), self.ap_clfs[i], dtype=int)
            else:
                X_detected      = X[aspect_mask]
                ap_for_detected = self.ap_clfs[i].predict(X_detected)

            y_pred[aspect_mask, i] = ap_for_detected
            # samples where ad_pred == 0 → y_pred[:, i] = 0 (already set)

        return y_pred

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'ad_clfs':    self.ad_clfs,
                'ap_clfs':    self.ap_clfs,
                'is_fitted':  self.is_fitted,
                'config': {
                    'max_features':  self.cfg.TFIDF_MAX_FEATURES,
                    'ngram_range':   self.cfg.TFIDF_NGRAM_RANGE,
                    'svm_c':         self.cfg.SVM_C,
                    'class_weight':  self.cfg.SVM_CLASS_WEIGHT,
                },
            }, f)
        print(f'  Model saved: {path}')

    @classmethod
    def load(cls, path: str, cfg: Config = None):
        if cfg is None:
            cfg = Config()
        obj = cls(cfg)
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj.vectorizer = state['vectorizer']
        obj.ad_clfs    = state['ad_clfs']
        obj.ap_clfs    = state['ap_clfs']
        obj.is_fitted  = state['is_fitted']
        return obj


# ══════════════════════════════════════════════════════════════════════════
# SYNC TO GOOGLE DRIVE
# ══════════════════════════════════════════════════════════════════════════
def sync_to_drive(local_dir: str, drive_dir: str):
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
        print(f'  [Drive] Synced {local_dir} → {drive_dir}')
    except Exception as e:
        print(f'  [Drive] Sync skipped: {e}')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(CFG.MODEL_DIR,   exist_ok=True)
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    print('\n' + '='*60)
    print('  TRAIN SVM + TF-IDF — ABSA (AD + AP)')
    print('='*60)

    # 1. Load data
    print('\n[1] Đọc dữ liệu...')
    train_texts, train_labels = load_csv_split(CFG.DATA_DIR, 'train')
    val_texts,   val_labels   = load_csv_split(CFG.DATA_DIR, 'val')
    test_texts,  test_labels  = load_csv_split(CFG.DATA_DIR, 'test')
    print(f'  Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}')

    # Gộp train + val để train final model (như thông lệ với SVM)
    all_train_texts  = train_texts + val_texts
    all_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    print(f'  Train + Val = {len(all_train_texts):,} samples (dùng cho final model)')

    # 2. Train trên train set, evaluate trên val set trước
    print('\n[2] Train trên train set, evaluate trên val set...')
    model_val = SVMTFIDFModel(CFG)
    t0 = time.time()
    model_val.fit(train_texts, train_labels)
    train_time = time.time() - t0
    print(f'\n  Thời gian train: {train_time:.1f}s')

    print('\n[3] Evaluate trên val set...')
    y_pred_val = model_val.predict(val_texts)
    val_metrics = evaluate_predictions(val_labels, y_pred_val)

    print(f'\n  ── Val Set Metrics ──────────────────────')
    print(f'  AD Precision : {val_metrics["ad_precision"]:.4f}')
    print(f'  AD Recall    : {val_metrics["ad_recall"]:.4f}')
    print(f'  AD F1        : {val_metrics["ad_f1"]:.4f}')
    print(f'  AP Precision : {val_metrics["ap_precision"]:.4f}')
    print(f'  AP Recall    : {val_metrics["ap_recall"]:.4f}')
    print(f'  AP F1        : {val_metrics["ap_f1"]:.4f}')
    print(f'  Average F1   : {val_metrics["avg_f1"]:.4f}')
    print(f'  ─────────────────────────────────────────')

    # 3. Train final model trên train + val, evaluate trên test
    print('\n[4] Train final model trên Train + Val...')
    model_final = SVMTFIDFModel(CFG)
    t0 = time.time()
    model_final.fit(all_train_texts, all_train_labels)
    print(f'  Thời gian train final: {time.time()-t0:.1f}s')

    print('\n[5] Evaluate trên test set...')
    y_pred_test = model_final.predict(test_texts)
    test_metrics = evaluate_predictions(test_labels, y_pred_test)

    print('\n' + '='*55)
    print('  KẾT QUẢ FINAL — SVM + TF-IDF (Test Set)')
    print('='*55)
    print(f'  AD Precision : {test_metrics["ad_precision"]:.4f}')
    print(f'  AD Recall    : {test_metrics["ad_recall"]:.4f}')
    print(f'  AD F1        : {test_metrics["ad_f1"]:.4f}')
    print(f'  AP Precision : {test_metrics["ap_precision"]:.4f}')
    print(f'  AP Recall    : {test_metrics["ap_recall"]:.4f}')
    print(f'  AP F1        : {test_metrics["ap_f1"]:.4f}')
    print(f'  Average F1   : {test_metrics["avg_f1"]:.4f}')
    print('='*55)

    print('\n  AD-F1 theo từng category (Test Set):')
    print(f'  {"Category":<30} {"Precision":>10} {"Recall":>8} {"F1":>8} {"Support":>8}')
    print('  ' + '-'*64)
    for cat in CATEGORIES:
        m = test_metrics['per_category'][cat]
        print(f'  {cat:<30} {m["precision"]:>10.4f} {m["recall"]:>8.4f} '
              f'{m["f1"]:>8.4f} {m["support"]:>8}')

    # 4. Save model
    model_path = os.path.join(CFG.MODEL_DIR, 'svm_model.pkl')
    model_final.save(model_path)
    sync_to_drive(CFG.MODEL_DIR, CFG.DRIVE_MODEL_DIR)

    # 5. Save results
    results_path = os.path.join(CFG.RESULTS_DIR, 'svm_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model':      'SVM+TF-IDF',
            'best_val':   val_metrics,
            'test':       test_metrics,
            'train_time': train_time,
            'config': {
                'max_features':  CFG.TFIDF_MAX_FEATURES,
                'ngram_range':   list(CFG.TFIDF_NGRAM_RANGE),
                'min_df':        CFG.TFIDF_MIN_DF,
                'sublinear_tf':  CFG.TFIDF_SUBLINEAR_TF,
                'svm_c':         CFG.SVM_C,
                'class_weight':  CFG.SVM_CLASS_WEIGHT,
                'train_samples': len(all_train_texts),
                'test_samples':  len(test_texts),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f'\n  Kết quả: {results_path}')
    print(f'  Model  : {model_path}')


if __name__ == '__main__':
    main()
