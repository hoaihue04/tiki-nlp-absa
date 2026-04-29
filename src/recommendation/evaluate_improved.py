#!/usr/bin/env python3
"""
evaluate_improved.py - Đánh giá hệ thống gợi ý hybrid cải tiến
============================================================
Tính các chỉ số Precision@K, Recall@K, NDCG@K
So sánh 3 phương pháp: CBF, ABSA, HYBRID (improved)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.improved_hybrid_recommender import (
    ImprovedHybridRecommender,
    ImprovedHybridConfig,
    ImprovedPhoBERTPredictor,
    extract_product_id_from_url,
)


def parse_k_values(raw: str) -> List[int]:
    """Parse k values từ command line"""
    return sorted(set(int(k.strip()) for k in raw.split(",") if k.strip()))


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """Tính Precision@K"""
    if k <= 0 or not ranked:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / k


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """Tính Recall@K"""
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: List[str], grade_map: Dict[str, float], k: int) -> float:
    """Tính DCG@K"""
    score = 0.0
    for idx, pid in enumerate(ranked[:k], start=1):
        rel = grade_map.get(pid, 0.0)
        if rel <= 0:
            continue
        score += (2.0 ** rel - 1.0) / math.log2(idx + 1.0)
    return score


def ndcg_at_k(ranked: List[str], grade_map: Dict[str, float], k: int) -> float:
    """Tính NDCG@K"""
    ideal_items = sorted(grade_map.items(), key=lambda x: x[1], reverse=True)
    ideal_ranked = [pid for pid, _ in ideal_items]
    ideal_dcg = dcg_at_k(ideal_ranked, grade_map, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(ranked, grade_map, k) / ideal_dcg


def build_co_review_relevance(
    query_product_id: str,
    recommender: ImprovedHybridRecommender,
    min_co_review: int = 1,
    boost_same_category: bool = True,
) -> Dict[str, float]:
    """
    Xây dựng ground truth relevance dựa trên co-review và cùng category
    """
    query_pid = str(query_product_id)
    
    # Đọc dữ liệu reviews
    df = recommender.reviews_df.copy()
    df["product_id"] = df["product_id"].astype(str)
    df["customer_id"] = df.get("customer_id", "").fillna("").astype(str)
    df = df[df["customer_id"].str.len() > 0]
    
    # Tìm khách hàng đã review query product
    query_customers = set(df[df["product_id"] == query_pid]["customer_id"].tolist())
    
    if not query_customers:
        return {}
    
    # Đếm số lượng co-review
    co_counts: Dict[str, int] = defaultdict(int)
    for cid in query_customers:
        for pid in df[df["customer_id"] == cid]["product_id"].tolist():
            if pid != query_pid:
                co_counts[pid] += 1
    
    # Boost sản phẩm cùng category
    if boost_same_category:
        query_meta = recommender._get_product_meta(query_pid)
        query_cat_lv3 = str(query_meta.get('category_lv3', ''))
        
        if query_cat_lv3 and query_cat_lv3 != 'nan':
            same_cat_products = recommender.products_df[
                recommender.products_df['category_lv3'].astype(str) == query_cat_lv3
            ]['product_id'].astype(str).tolist()
            
            for pid in same_cat_products:
                if pid != query_pid:
                    co_counts[pid] = max(co_counts.get(pid, 0), min_co_review)
    
    # Lọc theo threshold
    out = {pid: float(c) for pid, c in co_counts.items() if c >= min_co_review}
    
    # Thêm trọng số dựa trên rating
    for pid in out:
        meta = recommender._get_product_meta(pid)
        rating = float(meta.get('rating_average', 3) or 3)
        out[pid] = out[pid] * (0.5 + rating / 10.0)  # rating 5 → boost 1.0, rating 3 → boost 0.8
    
    return out


def evaluate_query(
    query_pid: str,
    recommender: ImprovedHybridRecommender,
    candidate_pool_size: int,
    reviews_source: str,
    relevance_func,
    relevance_kwargs: dict,
) -> Optional[Dict[str, Any]]:
    """
    Đánh giá một query cho cả 3 model
    """
    try:
        # Load reviews của query
        query_reviews = recommender.load_reviews(query_pid, reviews_source=reviews_source)
        query_reviews = recommender.preprocess_reviews(query_reviews)
        
        if len(query_reviews) < 2:
            print(f"  Skip: only {len(query_reviews)} reviews")
            return None
        
        # Lấy candidates
        candidates = recommender.get_candidates_by_category(query_pid)
        if len(candidates) < 10:
            print(f"  Skip: only {len(candidates)} candidates")
            return None
        
        # Tính CBF scores
        candidates = recommender.compute_cbf_scores(query_pid, candidates)
        
        # Tính ABSA scores
        candidates['absa_score'] = candidates['product_id'].apply(recommender._get_absa_cached)
        candidates['category_match_score'] = 1.0
        
        # Tính hybrid scores
        candidates, weights = recommender.compute_hybrid_scores(query_reviews, candidates)
        
        # Tạo ranking cho 3 model
        ranking_cols = {
            'CBF': 'cbf_score',
            'ABSA': 'absa_score',
            'HYBRID': 'hybrid_score'
        }
        
        ranks = {}
        for model_name, score_col in ranking_cols.items():
            ranked_df = candidates.sort_values(score_col, ascending=False)
            ranks[model_name] = ranked_df['product_id'].astype(str).tolist()
        
        # Lấy ground truth relevance
        grade_map = relevance_func(query_pid, recommender, **relevance_kwargs)
        relevant_set = set(grade_map.keys())
        
        if len(relevant_set) < 3:
            print(f"  Skip: only {len(relevant_set)} relevant items")
            return None
        
        return {
            "query_pid": query_pid,
            "n_reviews": len(query_reviews),
            "noise_ratio": float(query_reviews['noise_score'].mean()),
            "n_relevant": len(relevant_set),
            "weights": weights,
            "ranks": ranks,
            "grade_map": grade_map,
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def select_query_products(
    recommender: ImprovedHybridRecommender,
    num_queries: int,
    min_reviews: int,
    seed: int,
) -> List[str]:
    """Chọn ngẫu nhiên các sản phẩm có đủ review để làm query"""
    review_counts = recommender.reviews_df.groupby("product_id").size().reset_index(name="n_reviews")
    review_counts["product_id"] = review_counts["product_id"].astype(str)
    
    valid_products = set(recommender.products_df["product_id"].astype(str).tolist())
    eligible = review_counts[
        (review_counts["n_reviews"] >= min_reviews) & 
        (review_counts["product_id"].isin(valid_products))
    ]["product_id"].tolist()
    
    random.Random(seed).shuffle(eligible)
    return eligible[:num_queries]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Improved Hybrid Recommender")
    parser.add_argument("--num_queries", type=int, default=30, help="Số lượng query để đánh giá")
    parser.add_argument("--min_query_reviews", type=int, default=3, help="Số review tối thiểu cho query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k_values", type=str, default="5,10,20", help="Các giá trị K")
    parser.add_argument("--reviews_source", type=str, default="local", choices=["local", "auto", "live"])
    parser.add_argument("--candidate_pool", type=int, default=150)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_co_review", type=int, default=1)
    parser.add_argument("--boost_same_category", action="store_true", default=True)
    parser.add_argument("--category_level", type=str, default="lv3", choices=["lv2", "lv3", "id"])
    parser.add_argument("--output_root", type=str, default="results/improved_evaluation")
    
    args = parser.parse_args()
    k_values = parse_k_values(args.k_values)
    
    print("\n" + "="*70)
    print("📊 IMPROVED HYBRID RECOMMENDER EVALUATION")
    print("="*70)
    print(f"Config: {args.num_queries} queries, K={k_values}")
    print(f"Category level: {args.category_level}")
    print(f"Boost same category: {args.boost_same_category}")
    
    # Khởi tạo
    config = ImprovedHybridConfig(
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool,
        category_level=args.category_level,
    )
    
    predictor = ImprovedPhoBERTPredictor(max_len=config.max_len)
    recommender = ImprovedHybridRecommender(config, predictor)
    
    # Chọn query products
    query_ids = select_query_products(
        recommender,
        num_queries=args.num_queries,
        min_reviews=args.min_query_reviews,
        seed=args.seed,
    )
    print(f"\n📋 Selected {len(query_ids)} query products")
    
    # Đánh giá từng query
    results = []
    for idx, qid in enumerate(query_ids, 1):
        print(f"[{idx}/{len(query_ids)}] Evaluating {qid}...")
        
        eval_result = evaluate_query(
            query_pid=qid,
            recommender=recommender,
            candidate_pool_size=args.candidate_pool,
            reviews_source=args.reviews_source,
            relevance_func=build_co_review_relevance,
            relevance_kwargs={
                "min_co_review": args.min_co_review,
                "boost_same_category": args.boost_same_category,
            },
        )
        
        if eval_result:
            results.append(eval_result)
    
    if not results:
        raise RuntimeError("No valid queries evaluated")
    
    print(f"\n✅ Successfully evaluated {len(results)} queries")
    
    # Tính metrics
    per_query_rows = []
    
    for res in results:
        for model_name, ranked in res["ranks"].items():
            for k in k_values:
                row = {
                    "query_product_id": res["query_pid"],
                    "model": model_name,
                    "k": k,
                    "precision_at_k": precision_at_k(ranked, set(res["grade_map"].keys()), k),
                    "recall_at_k": recall_at_k(ranked, set(res["grade_map"].keys()), k),
                    "ndcg_at_k": ndcg_at_k(ranked, res["grade_map"], k),
                    "n_relevant": res["n_relevant"],
                    "n_reviews": res["n_reviews"],
                    "noise_ratio": res["noise_ratio"],
                }
                per_query_rows.append(row)
    
    # Tổng hợp
    per_query_df = pd.DataFrame(per_query_rows)
    summary_df = (
        per_query_df.groupby(["model", "k"], as_index=False)[["precision_at_k", "recall_at_k", "ndcg_at_k"]]
        .mean()
        .sort_values(["k", "model"])
    )
    
    # Lưu kết quả
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    per_query_df.to_csv(out_dir / "per_query_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    
    # Lưu JSON
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "num_queries": len(results),
                "k_values": k_values,
                "candidate_pool": args.candidate_pool,
                "category_level": args.category_level,
                "boost_same_category": args.boost_same_category,
            },
            "results": summary_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    
    # In kết quả
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print("\nMean metrics:")
    print(summary_df.to_string(index=False))
    
    print(f"\n📁 Results saved to: {out_dir}")


if __name__ == "__main__":
    main()