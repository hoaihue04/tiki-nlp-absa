#!/usr/bin/env python3
"""
Evaluate recommender protocol with Precision@K / Recall@K / NDCG@K.

Models compared:
- CBF-only
- ABSA-only
- Hybrid (dynamic alpha/beta)

Relevance sources:
- manual ground truth file
- co-review proxy from customer-product interactions in review data
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.hybrid_recommender import (  # noqa: E402
    HybridABSACBFRecommender,
    HybridConfig,
    PhoBERTABSAPredictor,
    compute_dynamic_alpha_beta,
    extract_product_id_from_url,
)


def parse_k_values(raw: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("k_values cannot be empty")
    return vals


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    if not topk:
        return 0.0
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / float(k)


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / float(len(relevant))


def dcg_at_k(ranked: Sequence[str], grade_map: Dict[str, float], k: int) -> float:
    score = 0.0
    for idx, pid in enumerate(ranked[:k], start=1):
        rel = float(grade_map.get(pid, 0.0))
        if rel <= 0:
            continue
        score += (2.0 ** rel - 1.0) / math.log2(idx + 1.0)
    return score


def ndcg_at_k(ranked: Sequence[str], grade_map: Dict[str, float], k: int) -> float:
    ideal_items = sorted(grade_map.items(), key=lambda x: x[1], reverse=True)
    ideal_ranked = [pid for pid, _ in ideal_items]
    ideal_dcg = dcg_at_k(ideal_ranked, grade_map, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(ranked, grade_map, k) / ideal_dcg


def load_queries_from_file(path: str) -> List[str]:
    df = pd.read_csv(path, dtype=str)
    cols = [c.lower() for c in df.columns]
    if "product_id" in cols:
        col = df.columns[cols.index("product_id")]
        return [str(x) for x in df[col].dropna().astype(str).tolist()]
    if "product_url" in cols:
        col = df.columns[cols.index("product_url")]
        urls = df[col].dropna().astype(str).tolist()
        out = []
        for u in urls:
            pid = extract_product_id_from_url(u)
            if pid:
                out.append(str(pid))
        return out
    raise ValueError("queries_file must have product_id or product_url column")


def select_query_products(
    recommender: HybridABSACBFRecommender,
    num_queries: int,
    min_query_reviews: int,
    seed: int,
) -> List[str]:
    review_counts = (
        recommender.reviews_df.groupby("product_id")["review_id"].count().reset_index(name="n_reviews")
    )
    review_counts["product_id"] = review_counts["product_id"].astype(str)
    valid_products = set(recommender.products_df["product_id"].astype(str).tolist())
    eligible = review_counts[
        (review_counts["n_reviews"] >= int(min_query_reviews))
        & (review_counts["product_id"].isin(valid_products))
    ]["product_id"].astype(str).tolist()

    random.Random(seed).shuffle(eligible)
    return eligible[: max(0, num_queries)]


def build_manual_relevance(gt_file: str) -> Dict[str, Dict[str, float]]:
    """
    Expected columns:
    - query_product_id
    - relevant_product_id
    Optional:
    - relevance_grade (default 1)
    """
    df = pd.read_csv(gt_file, dtype=str)
    needed = {"query_product_id", "relevant_product_id"}
    if not needed.issubset(set(df.columns)):
        raise ValueError("ground_truth_file must contain query_product_id, relevant_product_id")

    if "relevance_grade" not in df.columns:
        df["relevance_grade"] = "1"

    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        q = str(row["query_product_id"])
        p = str(row["relevant_product_id"])
        g = float(pd.to_numeric(row["relevance_grade"], errors="coerce"))
        if np.isnan(g):
            g = 1.0
        out[q][p] = max(g, out[q].get(p, 0.0))
    return dict(out)


def build_co_review_maps(
    reviews_df: pd.DataFrame,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build product->customers and customer->products maps."""
    df = reviews_df.copy()
    df["product_id"] = df["product_id"].astype(str)
    df["customer_id"] = df["customer_id"].fillna("").astype(str)
    df = df[df["customer_id"].str.len() > 0]

    product_to_customers: Dict[str, Set[str]] = defaultdict(set)
    customer_to_products: Dict[str, Set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        pid = row["product_id"]
        cid = row["customer_id"]
        product_to_customers[pid].add(cid)
        customer_to_products[cid].add(pid)

    return dict(product_to_customers), dict(customer_to_products)


def build_co_review_relevance(
    query_product_id: str,
    recommender: HybridABSACBFRecommender,
    product_to_customers: Dict[str, Set[str]],
    customer_to_products: Dict[str, Set[str]],
    min_co_review: int = 2,
    restrict_same_category: bool = True,
) -> Dict[str, float]:
    """
    Relevance grade = number of shared customers who reviewed query and candidate.
    """
    query_pid = str(query_product_id)
    customers = product_to_customers.get(query_pid, set())
    if not customers:
        return {}

    query_cat = ""
    qmeta = recommender.products_df[recommender.products_df["product_id"] == query_pid]
    if not qmeta.empty:
        query_cat = str(qmeta.iloc[0].get("category_id", ""))

    co_counts: Dict[str, int] = defaultdict(int)
    for cid in customers:
        for pid in customer_to_products.get(cid, set()):
            if pid == query_pid:
                continue
            co_counts[pid] += 1

    if restrict_same_category and query_cat:
        product_cat = recommender.products_df[["product_id", "category_id"]].drop_duplicates()
        cat_map = {
            str(r["product_id"]): str(r["category_id"])
            for _, r in product_cat.iterrows()
        }
        co_counts = {pid: c for pid, c in co_counts.items() if cat_map.get(str(pid), "") == query_cat}

    out = {str(pid): float(c) for pid, c in co_counts.items() if c >= int(min_co_review)}
    return out


def evaluate_query_for_all_models(
    query_pid: str,
    recommender: HybridABSACBFRecommender,
    cfg_eval: Dict[str, Any],
    absa_cache: Dict[str, float],
) -> Dict[str, Any]:
    query_reviews = recommender.load_reviews(query_pid, reviews_source=cfg_eval["reviews_source"])
    query_reviews = recommender.preprocess_reviews(query_reviews)
    quality = recommender._compute_quality_stats(query_reviews)

    alpha, beta, alpha_case1, alpha_case2 = compute_dynamic_alpha_beta(
        n_reviews=quality["n_reviews"],
        noise_ratio=quality["noise_ratio"],
        alpha_min=recommender.cfg.alpha_min,
        alpha_max=recommender.cfg.alpha_max,
        n_threshold=recommender.cfg.n_threshold,
        lambda_noise=recommender.cfg.lambda_noise,
    )

    cbf = recommender._compute_cbf_scores(query_pid)
    cands = cbf.head(int(cfg_eval["candidate_pool"])).copy()

    def _absa_score_for_product(pid: str) -> float:
        if pid in absa_cache:
            return absa_cache[pid]
        rv = recommender._load_local_reviews(pid)
        rv = recommender.preprocess_reviews(rv)
        s = recommender.compute_absa_score(rv, max_reviews=recommender.cfg.max_candidate_reviews)
        absa_cache[pid] = float(s["p_absa"])
        return absa_cache[pid]

    cands["absa_score"] = cands["product_id"].astype(str).apply(_absa_score_for_product)
    cands["hybrid_score"] = alpha * cands["absa_score"] + beta * cands["cbf_score"]

    ranks = {
        "CBF": cands.sort_values("cbf_score", ascending=False)["product_id"].astype(str).tolist(),
        "ABSA": cands.sort_values("absa_score", ascending=False)["product_id"].astype(str).tolist(),
        "HYBRID": cands.sort_values("hybrid_score", ascending=False)["product_id"].astype(str).tolist(),
    }

    return {
        "query_pid": str(query_pid),
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_case1": float(alpha_case1),
        "alpha_case2_before_clip": float(alpha_case2),
        "n_reviews": int(quality["n_reviews"]),
        "noise_ratio": float(quality["noise_ratio"]),
        "candidate_df": cands,
        "ranks": ranks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid recommender with P@K, R@K, NDCG@K")
    parser.add_argument("--queries_file", type=str, default=None, help="CSV with product_id or product_url")
    parser.add_argument("--num_queries", type=int, default=30)
    parser.add_argument("--min_query_reviews", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_values", type=str, default="5,10,20")
    parser.add_argument("--reviews_source", type=str, default="local", choices=["local", "auto", "live"])
    parser.add_argument("--candidate_pool", type=int, default=100, help="Candidate size before ranking")
    parser.add_argument("--top_k", type=int, default=20, help="K used inside recommender outputs")
    parser.add_argument("--preselect_cbf", type=int, default=100)
    parser.add_argument("--max_candidate_reviews", type=int, default=30)
    parser.add_argument("--alpha_min", type=float, default=0.10)
    parser.add_argument("--alpha_max", type=float, default=0.70)
    parser.add_argument("--n_threshold", type=int, default=10)
    parser.add_argument("--lambda_noise", type=float, default=0.80)
    parser.add_argument(
        "--relevance_source",
        type=str,
        default="co_review",
        choices=["co_review", "manual"],
    )
    parser.add_argument("--ground_truth_file", type=str, default=None, help="Required for manual relevance")
    parser.add_argument("--min_co_review", type=int, default=2)
    parser.add_argument("--restrict_same_category", action="store_true")
    parser.add_argument("--output_root", type=str, default="results/recommendation_eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    k_values = parse_k_values(args.k_values)

    cfg = HybridConfig(
        top_k=args.top_k,
        preselect_cbf=args.preselect_cbf,
        max_candidate_reviews=args.max_candidate_reviews,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_threshold=args.n_threshold,
        lambda_noise=args.lambda_noise,
    )

    predictor = PhoBERTABSAPredictor(max_len=cfg.max_len)
    recommender = HybridABSACBFRecommender(config=cfg, predictor=predictor)

    if args.queries_file:
        query_ids = load_queries_from_file(args.queries_file)
    else:
        query_ids = select_query_products(
            recommender=recommender,
            num_queries=args.num_queries,
            min_query_reviews=args.min_query_reviews,
            seed=args.seed,
        )
    query_ids = [str(q) for q in query_ids if str(q)]
    query_ids = list(dict.fromkeys(query_ids))

    if not query_ids:
        raise ValueError("No query products selected")

    manual_gt: Dict[str, Dict[str, float]] = {}
    if args.relevance_source == "manual":
        if not args.ground_truth_file:
            raise ValueError("--ground_truth_file is required when relevance_source=manual")
        manual_gt = build_manual_relevance(args.ground_truth_file)

    product_to_customers: Dict[str, Set[str]] = {}
    customer_to_products: Dict[str, Set[str]] = {}
    if args.relevance_source == "co_review":
        product_to_customers, customer_to_products = build_co_review_maps(recommender.reviews_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    absa_cache: Dict[str, float] = {}
    per_query_rows: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []

    for idx, qid in enumerate(query_ids, start=1):
        print(f"[{idx}/{len(query_ids)}] Evaluating query product {qid} ...")

        try:
            eval_pack = evaluate_query_for_all_models(
                query_pid=qid,
                recommender=recommender,
                cfg_eval={
                    "candidate_pool": args.candidate_pool,
                    "reviews_source": args.reviews_source,
                },
                absa_cache=absa_cache,
            )
        except Exception as e:
            print(f"  Skip query {qid}: {e}")
            continue

        if args.relevance_source == "manual":
            grade_map = manual_gt.get(str(qid), {})
        else:
            grade_map = build_co_review_relevance(
                query_product_id=str(qid),
                recommender=recommender,
                product_to_customers=product_to_customers,
                customer_to_products=customer_to_products,
                min_co_review=args.min_co_review,
                restrict_same_category=bool(args.restrict_same_category),
            )

        relevant_set = set(grade_map.keys())
        if not relevant_set:
            print(f"  Skip query {qid}: no relevant items from {args.relevance_source}")
            continue

        protocol_rows.append(
            {
                "query_product_id": str(qid),
                "n_relevant": int(len(relevant_set)),
                "alpha": eval_pack["alpha"],
                "beta": eval_pack["beta"],
                "n_reviews": eval_pack["n_reviews"],
                "noise_ratio": eval_pack["noise_ratio"],
                "relevance_source": args.relevance_source,
            }
        )

        for model_name, ranked in eval_pack["ranks"].items():
            for k in k_values:
                row = {
                    "query_product_id": str(qid),
                    "model": model_name,
                    "k": int(k),
                    "precision_at_k": precision_at_k(ranked, relevant_set, k),
                    "recall_at_k": recall_at_k(ranked, relevant_set, k),
                    "ndcg_at_k": ndcg_at_k(ranked, grade_map, k),
                    "n_relevant": int(len(relevant_set)),
                    "alpha": eval_pack["alpha"],
                    "beta": eval_pack["beta"],
                    "n_reviews": eval_pack["n_reviews"],
                    "noise_ratio": eval_pack["noise_ratio"],
                }
                per_query_rows.append(row)

    if not per_query_rows:
        raise RuntimeError("No evaluable queries produced metrics.")

    per_query_df = pd.DataFrame(per_query_rows)
    summary_df = (
        per_query_df.groupby(["model", "k"], as_index=False)[["precision_at_k", "recall_at_k", "ndcg_at_k"]]
        .mean()
        .sort_values(["k", "model"])
    )

    protocol_df = pd.DataFrame(protocol_rows)

    per_query_csv = out_dir / "per_query_metrics.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    protocol_csv = out_dir / "query_protocol.csv"
    summary_json = out_dir / "summary_metrics.json"

    per_query_df.to_csv(per_query_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    protocol_df.to_csv(protocol_csv, index=False, encoding="utf-8-sig")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "k_values": k_values,
                    "queries_count_input": len(query_ids),
                    "queries_count_used": int(per_query_df["query_product_id"].nunique()),
                    "relevance_source": args.relevance_source,
                    "candidate_pool": int(args.candidate_pool),
                    "reviews_source": args.reviews_source,
                },
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== EVALUATION DONE ===")
    print(f"Output dir: {out_dir}")
    print(f"Per-query metrics: {per_query_csv}")
    print(f"Summary metrics:   {summary_csv}")
    print(f"Protocol log:      {protocol_csv}")
    print("\nMean metrics (P@K, R@K, NDCG@K):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
