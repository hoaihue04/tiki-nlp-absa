#!/usr/bin/env python3
"""
run_improved_demo.py - Demo chạy hệ thống gợi ý hybrid cải tiến
"""

import argparse
import json
from pathlib import Path

from improved_hybrid_recommender import (
    ImprovedHybridRecommender,
    ImprovedHybridConfig,
    ImprovedPhoBERTPredictor,
    extract_product_id_from_url,
)


def main():
    parser = argparse.ArgumentParser(description="Improved Hybrid Recommender Demo")
    parser.add_argument("--product_url", type=str, required=True, help="Tiki product URL")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--category_level", type=str, default="lv3", choices=["lv2", "lv3", "id"])
    
    args = parser.parse_args()
    
    # Khởi tạo
    config = ImprovedHybridConfig(
        top_k=args.top_k,
        category_level=args.category_level,
    )
    
    predictor = ImprovedPhoBERTPredictor()
    recommender = ImprovedHybridRecommender(config, predictor)
    
    # Chạy
    result = recommender.run(product_url=args.product_url)
    
    # In kết quả
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION RESULTS")
    print("="*70)
    print(f"Query: {result['input']['product_name']}")
    print(f"Category: {result['input']['category_lv3']}")
    print(f"Weights: ABSA={result['weights']['absa']:.2f}, CBF={result['weights']['cbf']:.2f}, Cat={result['weights']['category']:.2f}")
    print("\nTop Recommendations:")
    print("-"*50)
    
    for i, item in enumerate(result['top_k'][:args.top_k], 1):
        print(f"{i:2d}. {item['name'][:60]}")
        print(f"    Score: {item['hybrid_score']:.4f} (ABSA={item['absa_score']:.3f}, CBF={item['cbf_score']:.3f})")
    
    print(f"\n📁 Full results: {result['output_dir']}")


if __name__ == "__main__":
    main()