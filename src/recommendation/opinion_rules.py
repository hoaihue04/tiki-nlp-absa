#!/usr/bin/env python3
"""
Opinion extraction rules for ABSA results.
Extract opinion targets from sentences based on aspect and sentiment.
"""

from typing import Dict, Any, Optional
import re


def extract_opinion_target(
    sentence: str,
    aspect: str,
    sentiment: str,
    confidence: float
) -> Optional[Dict[str, Any]]:
    """
    Extract the opinion phrase/target from a sentence based on aspect.
    
    Args:
        sentence: The original sentence
        aspect: The aspect category (e.g., "PRODUCT#QUALITY")
        sentiment: "positive", "negative", or "neutral"
        confidence: Model confidence score
    
    Returns:
        Dictionary with 'opinion' key, or None if no opinion found
    """
    if not sentence or not sentence.strip():
        return None
    
    sentence_lower = sentence.lower()
    
    # Common opinion patterns by aspect
    aspect_keywords = {
        "PRODUCT#QUALITY": ["chất lượng", "tốt", "dở", "ổn", "ngon", "kém", "tệ"],
        "PRODUCT#MATERIAL": ["chất liệu", "vải", "cotton", "nhựa", "cao su", "thun"],
        "PRODUCT#COMFORT": ["thoải mái", "dễ chịu", "thoáng", "dễ thở", "cộm", "ngứa"],
        "PRODUCT#SIZE": ["size", "kích thước", "to", "nhỏ", "vừa", "chật", "rộng"],
        "PRODUCT#DESIGN": ["thiết kế", "màu sắc", "kiểu dáng", "đẹp", "xấu", "dễ thương"],
        "PRODUCT#SAFETY": ["an toàn", "độc hại", "chất độc", "yên tâm", "nguy hiểm"],
        "PRODUCT#FUNCTION": ["tính năng", "chức năng", "công dụng", "tiện ích"],
        "PRODUCT#DURABILITY": ["bền", "chắc", "hỏng", "rách", "tốt", "lâu dài"],
        "PRODUCT#VALUE": ["đáng tiền", "xứng đáng", "rẻ", "đắt", "mắc", "hời"],
        "PRICE#AFFORDABILITY": ["giá", "rẻ", "đắt", "hợp lý", "phải chăng", "mắc"],
        "PRICE#DISCOUNT": ["giảm giá", "khuyến mãi", "sale", "deal", "voucher"],
        "DELIVERY#SPEED": ["giao nhanh", "giao chậm", "vận chuyển", "ship", "nhận hàng"],
        "DELIVERY#PACKAGING": ["đóng gói", "bao bì", "hộp", "túi", "bọc", "cẩn thận"],
        "DELIVERY#ACCURACY": ["đúng hàng", "sai hàng", "thiếu", "đủ", "chính xác"],
        "SELLER#SERVICE": ["nhân viên", "tư vấn", "hỗ trợ", "phục vụ", "thái độ"],
        "SELLER#RESPONSIVENESS": ["phản hồi", "trả lời", "nhanh", "chậm", "chat"],
        "SELLER#AUTHENTICITY": ["hàng thật", "hàng giả", "chính hãng", "fake", "nhái"]
    }
    
    # Get keywords for this aspect
    keywords = aspect_keywords.get(aspect, [])
    
    # Find the opinion phrase
    opinion = None
    
    # Try to extract phrase containing keyword
    for keyword in keywords:
        # Find sentences containing keyword
        pattern = r'([^.!?]*\b' + re.escape(keyword) + r'\b[^.!?]*)'
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            opinion = match.group(1).strip()
            # Limit length
            if len(opinion) > 100:
                opinion = opinion[:97] + "..."
            break
    
    # If no keyword found, take first 80 chars
    if not opinion:
        opinion = sentence[:80] + ("..." if len(sentence) > 80 else "")
    
    return {
        "opinion": opinion,
        "aspect": aspect,
        "sentiment": sentiment,
        "confidence": confidence
    }