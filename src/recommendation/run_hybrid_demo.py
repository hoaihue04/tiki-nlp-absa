#!/usr/bin/env python3
"""
Beginner-friendly entrypoint for hybrid recommender.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.hybrid_recommender import HybridABSACBFRecommender, HybridConfig, PhoBERTABSAPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hybrid ABSA + CBF recommendation demo")
    parser.add_argument("--product_url", type=str, default=None, help="Tiki product URL")
    parser.add_argument("--product_id", type=str, default=None, help="Product ID if you do not have URL")
    parser.add_argument(
        "--reviews_source",
        type=str,
        default="local",
        choices=["auto", "local", "live"],
        help="Start with local for stable beginner run",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = HybridConfig(top_k=args.top_k)
    predictor = PhoBERTABSAPredictor(max_len=cfg.max_len)
    system = HybridABSACBFRecommender(config=cfg, predictor=predictor)

    summary = system.run(
        product_url=args.product_url,
        product_id=args.product_id,
        reviews_source=args.reviews_source,
    )

    print("\nDone.")
    print(f"Output folder: {summary['outputs']['output_dir']}")
    print(f"Top-k file: {summary['outputs']['top_k_csv']}")
    print(f"Radar chart: {summary['outputs']['aspect_radar_chart']}")
    print(f"Weight chart: {summary['outputs']['alpha_beta_chart']}")


if __name__ == "__main__":
    main()
