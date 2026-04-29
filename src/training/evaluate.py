#!/usr/bin/env python3
"""
evaluate.py — Đánh giá tổng hợp tất cả mô hình ABSA
=====================================================
So sánh metrics của 3 mô hình:
    - BiLSTM-CRF : deep learning sequence model
    - PhoBERT    : transformer-based model
    - SVM+TF-IDF : baseline machine learning model

Metrics được tính:
    - AD F1  (Aspect Detection): binary, có/không có khía cạnh
    - AP F1  (Aspect Polarity) : sentiment cho các khía cạnh xuất hiện
    - Per-category breakdown   : phân tích từng khía cạnh riêng

Cách chạy:
    python src/training/evaluate.py              # đánh giá tất cả model
    python src/training/evaluate.py --model bilstm
    python src/training/evaluate.py --model phobert
    python src/training/evaluate.py --model svm

Phụ thuộc: scikit-learn, numpy, pandas, torch, transformers
"""

import json
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from sklearn.metrics import (f1_score, precision_score, recall_score)

# ── Label maps ────────────────────────────────────────────────────────────
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
SENT2IDX   = {'none': 0, 'positive': 1, 'neutral': 2, 'negative': 3}
IDX2SENT   = SENT_NAMES

DATA_DIR    = 'data/training'
RESULTS_DIR = 'results'


# ══════════════════════════════════════════════════════════════════════════
# CORE METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def compute_ad_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  per_category: bool = False) -> dict:
    """
    Tính Aspect Detection (AD) metrics.
    y_true, y_pred: [N, 17], values 0..3 (0=none)
    AD task: binary — present (≠0) vs absent (=0)
    """
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()

    metrics = {
        'ad_precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'ad_recall':    recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'ad_f1':        f1_score(y_true_flat, y_pred_flat, zero_division=0),
    }

    if per_category:
        cat_metrics = {}
        for i, cat in enumerate(CATEGORIES):
            yt = y_true_bin[:, i]
            yp = y_pred_bin[:, i]
            cat_metrics[cat] = {
                'precision': precision_score(yt, yp, zero_division=0),
                'recall':    recall_score(yt, yp, zero_division=0),
                'f1':        f1_score(yt, yp, zero_division=0),
                'support':   int(yt.sum()),
            }
        metrics['per_category_ad'] = cat_metrics

    return metrics


def compute_ap_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  per_class: bool = False) -> dict:
    """
    Tính Aspect Polarity (AP) metrics.
    Chỉ xét các (sample, category) pairs mà y_true ≠ 0 (truly present).
    Sentiment: 3-class micro-F1 {1=positive, 2=neutral, 3=negative}
    """
    present_mask = (y_true != 0)
    y_true_ap    = y_true[present_mask]
    y_pred_ap    = y_pred[present_mask]

    if len(y_true_ap) == 0:
        return {'ap_f1': 0.0, 'ap_precision': 0.0, 'ap_recall': 0.0}

    labels = [1, 2, 3]
    metrics = {
        'ap_precision': precision_score(y_true_ap, y_pred_ap, labels=labels,
                                        average='micro', zero_division=0),
        'ap_recall':    recall_score(y_true_ap, y_pred_ap, labels=labels,
                                     average='micro', zero_division=0),
        'ap_f1':        f1_score(y_true_ap, y_pred_ap, labels=labels,
                                 average='micro', zero_division=0),
    }

    if per_class:
        for sent_idx, sent_name in enumerate(SENT_NAMES[1:], start=1):
            yt_bin = (y_true_ap == sent_idx).astype(int)
            yp_bin = (y_pred_ap == sent_idx).astype(int)
            metrics[f'ap_{sent_name}_f1']      = f1_score(yt_bin, yp_bin, zero_division=0)
            metrics[f'ap_{sent_name}_support'] = int((y_true_ap == sent_idx).sum())

    return metrics


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = '') -> dict:
    """
    Tổng hợp tất cả metrics theo chuẩn bài báo VLSP.
    Trả về dict đầy đủ để in và lưu.
    """
    ad = compute_ad_f1(y_true, y_pred, per_category=True)
    ap = compute_ap_f1(y_true, y_pred, per_class=True)

    return {
        'model':        model_name,
        'ad_precision': ad['ad_precision'],
        'ad_recall':    ad['ad_recall'],
        'ad_f1':        ad['ad_f1'],
        'ap_precision': ap['ap_precision'],
        'ap_recall':    ap['ap_recall'],
        'ap_f1':        ap['ap_f1'],
        'avg_f1':       (ad['ad_f1'] + ap['ap_f1']) / 2,
        'per_category': ad.get('per_category_ad', {}),
        'per_sentiment': {
            k: v for k, v in ap.items()
            if k.startswith('ap_positive') or k.startswith('ap_neutral') or
               k.startswith('ap_negative')
        },
        'n_samples':    len(y_true),
    }


# ══════════════════════════════════════════════════════════════════════════
# LOAD PREDICTIONS — từng model inference
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth(data_dir: str = DATA_DIR) -> np.ndarray:
    """Đọc ground truth từ phobert_test.csv."""
    path = os.path.join(data_dir, 'phobert_test.csv')
    df   = pd.read_csv(path, dtype=str)
    label_cols = [f'label_{i}' for i in range(NUM_CATS)]
    y_true = df[label_cols].fillna(0).astype(int).values  # [N, 17]
    return y_true


def run_bilstm_inference(test_file: str, model_path: str,
                         vocab_path: str) -> Optional[np.ndarray]:
    """Chạy BiLSTM-CRF inference, trả về predictions [N, 17]."""
    try:
        sys.path.insert(0, '.')
        from src.training.train_bilstm import (
            BiLSTMDataset, BiLSTMCRFModel, collate_fn, CFG
        )
        from torch.utils.data import DataLoader

        device = torch.device('cpu')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        ckpt  = torch.load(model_path, map_location=device)
        model = BiLSTMCRFModel(len(vocab), CFG).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        ds     = BiLSTMDataset(test_file, vocab)
        loader = DataLoader(ds, batch_size=64, collate_fn=collate_fn)

        preds = []
        with torch.no_grad():
            for token_ids, _, mask, labels in loader:
                token_ids = token_ids.to(device)
                mask      = mask.to(device)
                logits, _ = model(token_ids, mask)    # [B, 17, 4]
                p = logits.argmax(dim=-1).cpu().numpy()  # [B, 17]
                preds.append(p)
        return np.concatenate(preds, axis=0)

    except Exception as e:
        print(f'  [BiLSTM inference] Lỗi: {e}')
        return None


def run_phobert_inference(csv_path: str, model_path: str) -> Optional[np.ndarray]:
    """Chạy PhoBERT inference, trả về predictions [N, 17]."""
    try:
        from transformers import AutoTokenizer
        sys.path.insert(0, '.')
        from src.training.train_phobert import PhoBERTDataset, PhoBERTASQP, CFG
        from torch.utils.data import DataLoader

        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt      = torch.load(model_path, map_location=device)
        model_cfg = ckpt.get('config', {})
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.get('model_name', 'vinai/phobert-base-v2'))

        model = PhoBERTASQP(CFG).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        ds     = PhoBERTDataset(csv_path, tokenizer)
        loader = DataLoader(ds, batch_size=32)

        preds = []
        with torch.no_grad():
            for batch in loader:
                iids   = batch['input_ids'].to(device)
                amsk   = batch['attention_mask'].to(device)
                logits = model(iids, amsk)              # [B, 17, 4]
                p = logits.argmax(dim=-1).cpu().numpy() # [B, 17]
                preds.append(p)
        return np.concatenate(preds, axis=0)

    except Exception as e:
        print(f'  [PhoBERT inference] Lỗi: {e}')
        return None


def run_svm_inference(csv_path: str, model_path: str) -> Optional[np.ndarray]:
    """
    Chạy SVM+TF-IDF inference, trả về predictions [N, 17].
    Không cần GPU — chạy trên CPU.
    """
    try:
        sys.path.insert(0, '.')
        # Import SVMTFIDFModel từ train_svm_tfidf.py
        from src.training.train_svm_tfidf import SVMTFIDFModel

        # Load model
        model = SVMTFIDFModel.load(model_path)

        # Đọc test CSV
        df    = pd.read_csv(csv_path, dtype=str)
        texts = df['text'].fillna('').tolist()

        # Predict
        y_pred = model.predict(texts)  # [N, 17]
        return y_pred

    except Exception as e:
        print(f'  [SVM inference] Lỗi: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════
# REPORT PRINTING
# ══════════════════════════════════════════════════════════════════════════
def print_metrics(metrics: dict):
    m = metrics
    print(f"\n{'─'*55}")
    print(f"  Model: {m['model']}")
    print(f"{'─'*55}")
    print(f"  Task AD (Aspect Detection):   Micro-F1")
    print(f"    Precision : {m['ad_precision']:.4f}")
    print(f"    Recall    : {m['ad_recall']:.4f}")
    print(f"    F1        : {m['ad_f1']:.4f}  ← chính")
    print(f"  Task AP (Aspect Polarity):    Micro-F1")
    print(f"    Precision : {m['ap_precision']:.4f}")
    print(f"    Recall    : {m['ap_recall']:.4f}")
    print(f"    F1        : {m['ap_f1']:.4f}  ← chính")
    print(f"  Average F1  : {m['avg_f1']:.4f}  ← tổng hợp")
    print(f"  Samples     : {m['n_samples']:,}")

    if m.get('per_sentiment'):
        print(f"\n  Sentiment breakdown (AP):")
        for k, v in m['per_sentiment'].items():
            if k.endswith('_f1'):
                name = k.replace('ap_', '').replace('_f1', '')
                support_key = k.replace('_f1', '_support')
                sup = m['per_sentiment'].get(support_key, '?')
                print(f"    {name:12s}: F1={v:.4f} (n={sup})")


def print_comparison(results: List[dict]):
    """In bảng so sánh giống bài báo."""
    print('\n' + '='*70)
    print('  SO SÁNH CÁC MÔ HÌNH (Test Set)')
    print('='*70)
    header = (f"  {'Model':<18} {'AD-P':>8} {'AD-R':>8} {'AD-F1':>8} "
              f"{'AP-P':>8} {'AP-R':>8} {'AP-F1':>8} {'Avg-F1':>8}")
    print(header)
    print('  ' + '-'*68)
    for m in results:
        if m is None:
            continue
        row = (f"  {m['model']:<18} "
               f"{m['ad_precision']:>8.4f} "
               f"{m['ad_recall']:>8.4f} "
               f"{m['ad_f1']:>8.4f} "
               f"{m['ap_precision']:>8.4f} "
               f"{m['ap_recall']:>8.4f} "
               f"{m['ap_f1']:>8.4f} "
               f"{m['avg_f1']:>8.4f}")
        print(row)
    print('='*70)

    valid = [m for m in results if m is not None]
    if valid:
        best = max(valid, key=lambda x: x['avg_f1'])
        print(f"\n  ★ Best model: {best['model']} (Avg F1 = {best['avg_f1']:.4f})")


def print_category_breakdown(metrics_list: List[dict]):
    """In bảng AD-F1 theo từng category cho mỗi model."""
    print('\n' + '='*80)
    print('  AD-F1 THEO TỪNG CATEGORY')
    print('='*80)

    model_names = [m['model'] for m in metrics_list if m and m.get('per_category')]
    col_w  = 12
    header = f"  {'Category':<28}" + ''.join(f"{n:>{col_w}}" for n in model_names)
    print(header)
    print('  ' + '-'*78)

    for cat in CATEGORIES:
        row = f"  {cat:<28}"
        for m in metrics_list:
            if m and m.get('per_category') and cat in m['per_category']:
                f1  = m['per_category'][cat]['f1']
                row += f"{f1:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình ABSA')
    parser.add_argument('--model', choices=['bilstm', 'phobert', 'svm', 'all'],
                        default='all',
                        help='Model cần đánh giá (mặc định: all)')
    parser.add_argument('--data_dir',    default=DATA_DIR)
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print('\n' + '='*60)
    print('  ĐÁNH GIÁ MÔ HÌNH ABSA — TIKI Baby Products')
    print('='*60)

    # Đọc ground truth
    y_true = load_ground_truth(args.data_dir)
    print(f'\nGround truth: {y_true.shape[0]:,} test samples, {y_true.shape[1]} categories')

    all_results = []
    eval_models = (['bilstm', 'phobert', 'svm'] if args.model == 'all'
                   else [args.model])

    for model_name in eval_models:
        print(f'\n[{model_name.upper()}] Đang inference...')
        y_pred = None

        # ── BiLSTM-CRF ──────────────────────────────────────────────────
        if model_name == 'bilstm':
            model_path = 'models/bilstm/best_model.pt'
            vocab_path = 'data/training/vocab.json'
            test_file  = 'data/training/bilstm_test.txt'
            if os.path.exists(model_path):
                y_pred = run_bilstm_inference(test_file, model_path, vocab_path)
            else:
                print(f'  Model chưa train: {model_path}')
                print(f'  Hãy chạy: python src/training/train_bilstm.py')

        # ── PhoBERT ─────────────────────────────────────────────────────
        elif model_name == 'phobert':
            model_path = 'models/phobert/best_model.pt'
            test_file  = 'data/training/phobert_test.csv'
            if os.path.exists(model_path):
                y_pred = run_phobert_inference(test_file, model_path)
            else:
                print(f'  Model chưa train: {model_path}')
                print(f'  Hãy chạy: python src/training/train_phobert.py')

        # ── SVM + TF-IDF ────────────────────────────────────────────────
        elif model_name == 'svm':
            model_path = 'models/svm/svm_model.pkl'
            test_file  = 'data/training/phobert_test.csv'  # dùng chung file CSV
            if os.path.exists(model_path):
                y_pred = run_svm_inference(test_file, model_path)
            else:
                print(f'  Model chưa train: {model_path}')
                print(f'  Hãy chạy: python src/training/train_svm_tfidf.py')

        # ── Tính metrics nếu có predictions ─────────────────────────────
        if y_pred is not None:
            min_n = min(len(y_true), len(y_pred))
            m = compute_full_metrics(y_true[:min_n], y_pred[:min_n],
                                     model_name=model_name.upper())
            print_metrics(m)
            all_results.append(m)

            # Lưu kết quả riêng
            out_path = os.path.join(args.results_dir, f'{model_name}_eval.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(_to_native(m), f, ensure_ascii=False, indent=2)
            print(f'  Saved: {out_path}')
        else:
            all_results.append(None)

    # ── So sánh tổng hợp ────────────────────────────────────────────────
    valid_results = [r for r in all_results if r is not None]
    if len(valid_results) > 1:
        print_comparison(valid_results)
        print_category_breakdown(valid_results)

    # ── Lưu báo cáo tổng hợp ────────────────────────────────────────────
    if valid_results:
        report_path = os.path.join(args.results_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(_to_native({'models': valid_results}), f,
                      ensure_ascii=False, indent=2)
        print(f'\n  Báo cáo tổng hợp: {report_path}')

        report_txt = os.path.join(args.results_dir, 'evaluation_report.txt')
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write('=== EVALUATION REPORT — TIKI ABSA ===\n\n')
            for m in valid_results:
                f.write(f"Model: {m['model']}\n")
                f.write(f"  AD Precision : {m['ad_precision']:.4f}\n")
                f.write(f"  AD Recall    : {m['ad_recall']:.4f}\n")
                f.write(f"  AD F1        : {m['ad_f1']:.4f}\n")
                f.write(f"  AP Precision : {m['ap_precision']:.4f}\n")
                f.write(f"  AP Recall    : {m['ap_recall']:.4f}\n")
                f.write(f"  AP F1        : {m['ap_f1']:.4f}\n")
                f.write(f"  Avg F1       : {m['avg_f1']:.4f}\n\n")
        print(f'  Text report  : {report_txt}')


# ══════════════════════════════════════════════════════════════════════════
# UTILS
# ══════════════════════════════════════════════════════════════════════════
def _to_native(obj):
    """Chuyển numpy types về Python native để JSON serializable."""
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, dict):        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_to_native(v) for v in obj]
    return obj


if __name__ == '__main__':
    main()
