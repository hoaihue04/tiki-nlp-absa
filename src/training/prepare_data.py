#!/usr/bin/env python3
"""
prepare_data.py — Chuẩn bị dữ liệu cho BiLSTM-CRF, PhoBERT, ViT5
==================================================================
Cách chạy (từ thư mục gốc dự án):
    python src/training/prepare_data.py

Output (data/training/):
    bilstm_{train,val,test}.txt   — BIO sequence format
    phobert_{train,val,test}.csv  — text + 17 label columns
    vit5_{train,val,test}.csv     — text + target string
    vocab.json                    — từ điển cho BiLSTM
    label_map.json                — ánh xạ category/sentiment
    report_split_data.txt         — thống kê phân tách

Chiến lược xử lý mất cân bằng (KHÔNG dùng class weights):
    - Focal Loss được tích hợp trong các file train_*.py
    - Oversampling: neutral x6, negative x2 (có word dropout)
    - Oversampling chỉ áp dụng trên tập TRAIN để tránh data leakage
"""

import json
import os
import csv
import re
import random
import copy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

# ── Seed ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Danh sách 17 khía cạnh (thứ tự cố định) ───────────────────────────────
CATEGORIES = [
    'PRODUCT#QUALITY',       # 0
    'DELIVERY#SPEED',        # 1
    'DELIVERY#PACKAGING',    # 2
    'PRICE#AFFORDABILITY',   # 3
    'SELLER#SERVICE',        # 4
    'PRODUCT#FUNCTION',      # 5
    'PRODUCT#COMFORT',       # 6
    'PRODUCT#DESIGN',        # 7
    'DELIVERY#ACCURACY',     # 8
    'PRODUCT#DURABILITY',    # 9
    'PRODUCT#SAFETY',        # 10
    'SELLER#AUTHENTICITY',   # 11
    'PRODUCT#MATERIAL',      # 12
    'PRODUCT#SIZE',          # 13
    'PRODUCT#VALUE',         # 14
    'PRICE#DISCOUNT',        # 15
    'SELLER#RESPONSIVENESS', # 16
]
NUM_CATS   = len(CATEGORIES)
CAT2IDX    = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT    = CATEGORIES  # list index = category index

# Sentiment encoding: 0=none, 1=positive, 2=neutral, 3=negative
SENT2IDX   = {'none': 0, 'positive': 1, 'neutral': 2, 'negative': 3}
IDX2SENT   = ['none', 'positive', 'neutral', 'negative']

# ViT5 output format: "cat_vn:sent_vn | ..."
# Vi du: "toc-do-giao-hang:tot | chat-luong-san-pham:te"
SENT2VIT5 = {'positive': 'tot', 'neutral': 'tam', 'negative': 'te'}
VIT5_CATEGORIES_VN = {
    'PRODUCT#QUALITY':       'chat-luong-san-pham',
    'DELIVERY#SPEED':        'toc-do-giao-hang',
    'DELIVERY#PACKAGING':    'dong-goi-giao-hang',
    'PRICE#AFFORDABILITY':   'gia-ca-san-pham',
    'SELLER#SERVICE':        'dich-vu-nguoi-ban',
    'PRODUCT#FUNCTION':      'chuc-nang-san-pham',
    'PRODUCT#COMFORT':       'su-thoai-mai',
    'PRODUCT#DESIGN':        'thiet-ke-san-pham',
    'DELIVERY#ACCURACY':     'do-chinh-xac-giao-hang',
    'PRODUCT#DURABILITY':    'do-ben-san-pham',
    'PRODUCT#SAFETY':        'an-toan-san-pham',
    'SELLER#AUTHENTICITY':   'hang-chinh-hang',
    'PRODUCT#MATERIAL':      'chat-lieu-san-pham',
    'PRODUCT#SIZE':          'kich-thuoc-san-pham',
    'PRODUCT#VALUE':         'gia-tri-san-pham',
    'PRICE#DISCOUNT':        'giam-gia-san-pham',
    'SELLER#RESPONSIVENESS': 'phan-hoi-nguoi-ban',
}

# BIO tags cho aspect term extraction (7 tags)
BIO_TAGS = ['O',
            'B-positive', 'I-positive',
            'B-neutral',  'I-neutral',
            'B-negative', 'I-negative']
BIO2IDX  = {t: i for i, t in enumerate(BIO_TAGS)}

# ── Paths ─────────────────────────────────────────────────────────────────
INPUT_FILE = 'data/processed/asqp_annotated.jsonl'
OUTPUT_DIR = 'data/training'

# ── Tỷ lệ phân tách ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Oversampling config ───────────────────────────────────────────────────
NEUTRAL_OVERSAMPLE  = 6    # nhân 6 lần cho neutral
NEGATIVE_OVERSAMPLE = 2    # nhân 2 lần cho negative
WORD_DROPOUT_RATE   = 0.10 # xác suất drop từ khi augment


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f'[LOAD] {len(data):,} reviews từ {path}')
    return data


# ══════════════════════════════════════════════════════════════════════════
# 2. XÂY DỰNG LABEL VECTOR
# ══════════════════════════════════════════════════════════════════════════
def build_label_vector(quadruples: List[dict]) -> List[int]:
    """
    Chuyển quadruples → vector 17 chiều.
    Nếu 1 category xuất hiện nhiều lần với sentiment khác nhau,
    ưu tiên theo độ nghiêm trọng: negative > neutral > positive.
    """
    PRIORITY = {'negative': 3, 'neutral': 2, 'positive': 1, 'none': 0}
    vec = [0] * NUM_CATS
    for q in quadruples:
        cat  = q.get('aspect_category', '').strip()
        sent = q.get('sentiment', 'positive').lower().strip()
        if cat not in CAT2IDX:
            continue
        idx = CAT2IDX[cat]
        cur_priority = PRIORITY.get(IDX2SENT[vec[idx]], 0)
        new_priority = PRIORITY.get(sent, 0)
        if new_priority > cur_priority:
            vec[idx] = SENT2IDX.get(sent, 0)
    return vec


# ══════════════════════════════════════════════════════════════════════════
# 3. TOKENIZE ĐƠN GIẢN
# ══════════════════════════════════════════════════════════════════════════
def tokenize(text: str) -> List[str]:
    """
    Tokenize theo khoảng trắng + tách dấu câu.
    Hỗ trợ underthesea nếu cài đặt (bật bằng USE_UNDERTHESEA=True).
    """
    text = text.lower().strip()
    text = re.sub(r'([,\.!?;:()\[\]{}"\'])', r' \1 ', text)
    tokens = text.split()
    return tokens if tokens else ['<unk>']


# ══════════════════════════════════════════════════════════════════════════
# 4. XÂY DỰNG BIO SEQUENCE cho BiLSTM-CRF
# ══════════════════════════════════════════════════════════════════════════
def find_span_in_tokens(tokens: List[str], term: str) -> Tuple[int, int]:
    """Tìm vị trí bắt đầu/kết thúc của term trong tokens."""
    term_toks = tokenize(term)
    n = len(term_toks)
    if n == 0:
        return -1, -1
    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == term_toks:
            return i, i + n
    return -1, -1


def build_bio_sequence(tokens: List[str], quadruples: List[dict]) -> List[str]:
    """Xây dựng BIO tags cho aspect terms trong câu."""
    bio = ['O'] * len(tokens)
    tagged = set()  # tránh overlap
    for q in quadruples:
        term = q.get('aspect_term', '').strip()
        sent = q.get('sentiment', 'positive').lower().strip()
        if not term or sent not in ('positive', 'neutral', 'negative'):
            continue
        start, end = find_span_in_tokens(tokens, term)
        if start == -1:
            continue
        # Bỏ qua nếu span đã được tag
        if any(j in tagged for j in range(start, end)):
            continue
        bio[start] = f'B-{sent}'
        for j in range(start + 1, end):
            bio[j] = f'I-{sent}'
        tagged.update(range(start, end))
    return bio


# ══════════════════════════════════════════════════════════════════════════
# 5. AUGMENTATION (word dropout)
# ══════════════════════════════════════════════════════════════════════════
def word_dropout(text: str, rate: float = WORD_DROPOUT_RATE) -> str:
    """Randomly drop một số từ trong text để tạo augmented sample."""
    tokens = text.split()
    if len(tokens) <= 4:
        return text  # quá ngắn, không dropout
    kept = [t for t in tokens if random.random() > rate]
    if len(kept) < 3:
        return text
    return ' '.join(kept)


def word_swap(text: str) -> str:
    """Hoán vị ngẫu nhiên 1-2 cặp từ liền kề."""
    tokens = text.split()
    if len(tokens) < 4:
        return text
    n_swaps = max(1, len(tokens) // 20)
    for _ in range(n_swaps):
        i = random.randint(0, len(tokens) - 2)
        tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
    return ' '.join(tokens)


# ══════════════════════════════════════════════════════════════════════════
# 6. OVERSAMPLING MINORITY CLASSES
# ══════════════════════════════════════════════════════════════════════════
def oversample_minority(samples: List[dict]) -> List[dict]:
    """
    Duplicate samples chứa neutral/negative với word augmentation.
    Không dùng class weights — dùng oversampling thay thế.
    """
    aug_samples = list(samples)

    neutral_samples  = [s for s in samples if 2 in s['labels']]
    negative_samples = [s for s in samples if 3 in s['labels']]

    print(f'  Neutral samples:  {len(neutral_samples):,}')
    print(f'  Negative samples: {len(negative_samples):,}')
    print(f'  Positive samples: {len(samples) - len(neutral_samples):,} (ước tính)')

    # Oversample NEUTRAL: x6 (lớp thiểu số nhất)
    augment_fns = [
        lambda t: word_dropout(t, rate=0.10),
        lambda t: word_dropout(t, rate=0.08),
        lambda t: word_swap(t),
        lambda t: word_dropout(word_swap(t), rate=0.06),
        lambda t: word_dropout(t, rate=0.12),
    ]
    for rep_idx in range(NEUTRAL_OVERSAMPLE - 1):
        fn = augment_fns[rep_idx % len(augment_fns)]
        for s in neutral_samples:
            aug = copy.deepcopy(s)
            aug['text'] = fn(s['text'])
            aug['is_augmented'] = True
            aug_samples.append(aug)

    # Oversample NEGATIVE: x2
    for s in negative_samples:
        aug = copy.deepcopy(s)
        aug['text'] = word_dropout(s['text'], rate=0.08)
        aug['is_augmented'] = True
        aug_samples.append(aug)

    random.shuffle(aug_samples)
    print(f'  Sau oversampling: {len(aug_samples):,} samples')
    return aug_samples


# ══════════════════════════════════════════════════════════════════════════
# 7. XÂY DỰNG TỪ ĐIỂN cho BiLSTM
# ══════════════════════════════════════════════════════════════════════════
def build_vocab(samples: List[dict], min_freq: int = 2,
                max_size: int = 40000) -> Dict[str, int]:
    """Xây dựng từ điển từ tập training."""
    counter = Counter()
    for s in samples:
        counter.update(tokenize(s['text']))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    print(f'  Kích thước từ điển: {len(vocab):,} từ')
    return vocab


# ══════════════════════════════════════════════════════════════════════════
# 8. PHÂN TÁCH DỮ LIỆU (stratified)
# ══════════════════════════════════════════════════════════════════════════
def stratified_split(data: List[dict]) -> Tuple[List, List, List]:
    """
    Phân tách train/val/test theo phân bố category.
    Stratify bằng combination của categories hiện diện.
    """
    def get_strat_key(s):
        present_cats = sorted([i for i, l in enumerate(s['labels']) if l != 0])
        if not present_cats:
            return 'empty'
        # Dùng 2 cat đầu để stratify
        key = '_'.join(map(str, present_cats[:2]))
        return key

    strat_keys   = [get_strat_key(d) for d in data]
    key_counts   = Counter(strat_keys)
    final_keys   = [k if key_counts[k] >= 4 else 'other' for k in strat_keys]

    indices      = list(range(len(data)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=final_keys,
        random_state=SEED
    )

    temp_keys    = [final_keys[i] for i in temp_idx]
    temp_counts  = Counter(temp_keys)
    temp_final   = [k if temp_counts[k] >= 2 else 'other' for k in temp_keys]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_final,
        random_state=SEED
    )

    return ([data[i] for i in train_idx],
            [data[i] for i in val_idx],
            [data[i] for i in test_idx])


# ══════════════════════════════════════════════════════════════════════════
# 9. EXPORT FORMATS
# ══════════════════════════════════════════════════════════════════════════
def export_bilstm(samples: List[dict], path: str):
    """
    BIO format:
        # labels: l0 l1 ... l16
        token1 \\t bio_tag1
        token2 \\t bio_tag2
        ...
        <blank line>
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            tokens    = tokenize(s['text'])
            bio_tags  = build_bio_sequence(tokens, s.get('quadruples', []))
            label_str = ' '.join(map(str, s['labels']))
            f.write(f'# labels: {label_str}\n')
            for tok, tag in zip(tokens, bio_tags):
                f.write(f'{tok}\t{tag}\n')
            f.write('\n')
    print(f'  BiLSTM → {path} ({len(samples):,} samples)')


def export_phobert(samples: List[dict], path: str):
    """CSV: text, label_0, ..., label_16"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = ['text'] + [f'label_{i}' for i in range(NUM_CATS)]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        for s in samples:
            writer.writerow([s['text']] + s['labels'])
    print(f'  PhoBERT → {path} ({len(samples):,} samples)')


def export_vit5(samples: List[dict], path: str):
    """
    CSV: text, target
    target: "toc-do-giao-hang:tot | chat-luong-san-pham:te"
    hoac "none" neu khong co aspect nao
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['text', 'target'])
        for s in samples:
            parts = []
            for i, lbl in enumerate(s['labels']):
                if lbl != 0:
                    cat_code = VIT5_CATEGORIES_VN[IDX2CAT[i]]
                    sent_vn  = SENT2VIT5[IDX2SENT[lbl]]
                    parts.append(f'{cat_code}:{sent_vn}')
            target = ' | '.join(parts) if parts else 'none'
            writer.writerow([s['text'], target])
    print(f'  ViT5 → {path} ({len(samples):,} samples)')


# ══════════════════════════════════════════════════════════════════════════
# 10. BÁO CÁO THỐNG KÊ
# ══════════════════════════════════════════════════════════════════════════
def print_split_stats(name: str, samples: List[dict]):
    cat_counts  = Counter()
    sent_counts = Counter()
    multi_aspect = 0
    for s in samples:
        n_aspects = sum(1 for l in s['labels'] if l != 0)
        if n_aspects > 1:
            multi_aspect += 1
        for i, lbl in enumerate(s['labels']):
            if lbl != 0:
                cat_counts[IDX2CAT[i]] += 1
                sent_counts[IDX2SENT[lbl]] += 1
    print(f'\n  [{name}] {len(samples):,} samples | '
          f'multi-aspect: {multi_aspect:,} ({multi_aspect/max(len(samples),1)*100:.1f}%)')
    print(f'  Sentiment: {dict(sent_counts)}')
    print(f'  Top-5 categories: {dict(cat_counts.most_common(5))}')


def write_report(train, val, test, path: str):
    lines = ['=== DATA SPLIT REPORT ===\n']

    def section(name, samples):
        lines.append(f'\n[{name}] {len(samples):,} samples')
        cat_c = Counter()
        sent_c = Counter()
        for s in samples:
            for i, l in enumerate(s['labels']):
                if l != 0:
                    cat_c[IDX2CAT[i]] += 1
                    sent_c[IDX2SENT[l]] += 1
        lines.append(f'  Sentiment: {dict(sent_c)}')
        lines.append('  Categories:')
        for cat, cnt in cat_c.most_common():
            lines.append(f'    {cat}: {cnt}')

    section('TRAIN (after oversample)', train)
    section('VAL', val)
    section('TEST', test)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n  Report → {path}')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\n' + '='*60)
    print('  CHUẨN BỊ DỮ LIỆU ABSA — TIKI Baby Products')
    print('='*60)

    # 1. Load
    print('\n[1] Đọc dữ liệu annotation...')
    raw = load_jsonl(INPUT_FILE)

    # 2. Convert thành structured samples
    print('\n[2] Xây dựng label vectors...')
    samples = []
    skipped = 0
    for d in raw:
        text = d.get('text', '').strip()
        if len(text) < 3:
            skipped += 1
            continue
        samples.append({
            'review_id':    d.get('review_id', ''),
            'text':         text,
            'labels':       build_label_vector(d.get('quadruples', [])),
            'quadruples':   d.get('quadruples', []),
            'is_augmented': False,
        })
    print(f'  Hợp lệ: {len(samples):,} | Bỏ qua: {skipped}')

    # Thống kê ban đầu
    all_sent = Counter()
    for s in samples:
        for lbl in s['labels']:
            if lbl != 0:
                all_sent[IDX2SENT[lbl]] += 1
    print(f'  Phân bố sentiment: {dict(all_sent)}')

    # 3. Phân tách TRƯỚC khi oversample (tránh data leakage)
    print('\n[3] Phân tách train/val/test (stratified)...')
    train, val, test = stratified_split(samples)
    print(f'  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}')

    print_split_stats('TRAIN (gốc)', train)
    print_split_stats('VAL', val)
    print_split_stats('TEST', test)

    # 4. Oversample CHỈ trên train
    print('\n[4] Oversampling tập TRAIN...')
    train_aug = oversample_minority(train)

    # 5. Build vocab từ train (augmented)
    print('\n[5] Xây dựng từ điển cho BiLSTM...')
    vocab = build_vocab(train_aug)

    # 6. Export BiLSTM
    print('\n[6] Xuất BiLSTM format...')
    export_bilstm(train_aug, f'{OUTPUT_DIR}/bilstm_train.txt')
    export_bilstm(val,       f'{OUTPUT_DIR}/bilstm_val.txt')
    export_bilstm(test,      f'{OUTPUT_DIR}/bilstm_test.txt')

    # 7. Export PhoBERT
    print('\n[7] Xuất PhoBERT format...')
    export_phobert(train_aug, f'{OUTPUT_DIR}/phobert_train.csv')
    export_phobert(val,       f'{OUTPUT_DIR}/phobert_val.csv')
    export_phobert(test,      f'{OUTPUT_DIR}/phobert_test.csv')

    # 8. Export ViT5
    print('\n[8] Xuất ViT5 format...')
    export_vit5(train_aug, f'{OUTPUT_DIR}/vit5_train.csv')
    export_vit5(val,       f'{OUTPUT_DIR}/vit5_val.csv')
    export_vit5(test,      f'{OUTPUT_DIR}/vit5_test.csv')

    # 9. Lưu metadata
    print('\n[9] Lưu metadata...')
    label_map = {
        'categories':       CATEGORIES,
        'cat2idx':          CAT2IDX,
        'idx2cat':          IDX2CAT,
        'sentiments':       IDX2SENT,
        'sent2idx':         SENT2IDX,
        'bio_tags':         BIO_TAGS,
        'bio2idx':          BIO2IDX,
        'vit5_sent_map':    SENT2VIT5,
        'vit5_categories':  VIT5_CATEGORIES_VN,
        'num_categories':   NUM_CATS,
    }
    with open(f'{OUTPUT_DIR}/label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    with open(f'{OUTPUT_DIR}/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 10. Báo cáo
    print('\n[10] Viết báo cáo thống kê...')
    write_report(train_aug, val, test, f'{OUTPUT_DIR}/report_split_data.txt')

    print('\n' + '='*60)
    print('  HOÀN THÀNH! Dữ liệu đã sẵn sàng tại data/training/')
    print('='*60)
    print('\nBước tiếp theo:')
    print('  python src/training/train_bilstm.py')
    print('  python src/training/train_phobert.py')
    print('  python src/training/train_vit5.py')


if __name__ == '__main__':
    main()
