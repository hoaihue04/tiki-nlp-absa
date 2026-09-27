from __future__ import annotations

from typing import List

CATEGORIES: List[str] = [
    "PRODUCT#QUALITY",
    "DELIVERY#SPEED",
    "DELIVERY#PACKAGING",
    "PRICE#AFFORDABILITY",
    "SELLER#SERVICE",
    "PRODUCT#FUNCTION",
    "PRODUCT#COMFORT",
    "PRODUCT#DESIGN",
    "DELIVERY#ACCURACY",
    "PRODUCT#DURABILITY",
    "PRODUCT#SAFETY",
    "SELLER#AUTHENTICITY",
    "PRODUCT#MATERIAL",
    "PRODUCT#SIZE",
    "PRODUCT#VALUE",
    "PRICE#DISCOUNT",
    "SELLER#RESPONSIVENESS",
]

ASPECT_REQUIRED_KEYWORDS = {
    "PRODUCT#QUALITY": ["chất lượng","hàng","sản phẩm","tốt","ngon","dở","tệ","chất","ổn","ok","kém","tuyệt"],
    "PRODUCT#MATERIAL": ["chất liệu","vải","cotton","len","sợi","da","nhựa","lông","nỉ","polyester","mỏng","dày"],
    "PRODUCT#COMFORT": ["thoải mái","dễ chịu","mềm","êm","khó chịu","cứng","cảm giác","thoáng","ấm"],
    "PRODUCT#SIZE": ["size","kích thước","to","nhỏ","vừa","chật","rộng","lớn","bé","cỡ","fit","vừa vặn","vừa người"],
    "PRODUCT#DESIGN": ["thiết kế","mẫu","kiểu dáng","đẹp","xinh","xấu","màu","màu sắc","họa tiết","hoa văn","kiểu"],
    "PRODUCT#SAFETY": ["an toàn","bảo vệ","nguy hiểm","độc hại","chất độc","bpa","an toàn cho bé","không độc"],
    "PRODUCT#FUNCTION": ["chức năng","công dụng","dùng","sử dụng","tiện","hữu ích","tác dụng","tiện lợi","dễ dùng"],
    "PRODUCT#DURABILITY": ["bền","chắc","hỏng","rách","xước","mòn","lâu","chắc chắn","bền bỉ"],
    "PRODUCT#VALUE": ["giá trị","đáng tiền","tiền nào","xứng đáng","hợp lý","đáng mua","xứng"],
    "PRICE#AFFORDABILITY": ["giá","rẻ","đắt","mắc","tiền","bao nhiêu","giá cả","giá tiền","giá bán"],
    "PRICE#DISCOUNT": ["giảm giá","khuyến mãi","sale","ưu đãi","voucher","giảm","freeship","coupon","mã giảm"],
    "DELIVERY#SPEED": ["giao hàng","nhanh","chậm","vận chuyển","ship","giao","giao nhanh","nhận hàng","giao chậm"],
    "DELIVERY#PACKAGING": ["đóng gói","bao bì","hộp","túi","bọc","kỹ","cẩn thận","thùng","đóng hàng","bao bì"],
    "DELIVERY#ACCURACY": ["chính xác","sai","thiếu","thừa","đúng","lộn","giao đúng","đủ hàng","giao sai","giao nhầm"],
    "SELLER#SERVICE": ["shop","người bán","tư vấn","nhiệt tình","hỗ trợ","chăm sóc","thái độ","dịch vụ"],
    "SELLER#RESPONSIVENESS": ["phản hồi","trả lời","chat","liên lạc","nhanh nhẹn","rep","reply","hồi âm"],
    "SELLER#AUTHENTICITY": ["chính hãng","hàng thật","fake","nhái","xịn","giả","tem mác","hàng giả","hàng nhái"],
}

ASPECT_FORBIDDEN_KEYWORDS = {
    "PRODUCT#SIZE": ["giao","ship","vận chuyển","giá","rẻ","đắt","shop","người bán","đóng gói","phản hồi"],
    "PRODUCT#QUALITY": ["giao","ship","vận chuyển","size","kích thước","shop","giảm giá","sale"],
    "PRODUCT#MATERIAL": ["giao","ship","size","giá rẻ","giá đắt","shop","giao hàng"],
    "PRODUCT#COMFORT": ["giao","ship","giá","size","kích thước","đóng gói"],
    "PRODUCT#DESIGN": ["giao","ship","giá","size","kích thước","đóng gói","phản hồi"],
    "PRODUCT#FUNCTION": ["giao","ship","size","giá","đóng gói"],
    "PRODUCT#DURABILITY": ["giao","ship","size","giá","đóng gói"],
    "PRODUCT#VALUE": ["giao","ship","đóng gói","size"],
    "DELIVERY#SPEED": ["chất lượng","chất liệu","size","thiết kế","màu","đẹp","xấu","mềm","cứng","bền"],
    "DELIVERY#PACKAGING": ["chất lượng","size","thiết kế","màu","mềm","cứng"],
    "DELIVERY#ACCURACY": ["chất lượng","size","thiết kế","màu","mềm"],
    "PRICE#AFFORDABILITY": ["giao","ship","đóng gói","chất liệu","size","thiết kế"],
    "PRICE#DISCOUNT": ["giao","ship","chất lượng","size","chất liệu"],
    "SELLER#SERVICE": ["giao","ship","vận chuyển","size","chất liệu","thiết kế"],
    "SELLER#RESPONSIVENESS": ["giao","ship","size","chất liệu","thiết kế"],
    "SELLER#AUTHENTICITY": ["giao","ship","size","chất liệu","thiết kế"],
}

ASPECT_DISPLAY_NAMES = {
    "PRODUCT#SIZE": "Kích thước / độ vừa vặn",
    "PRODUCT#QUALITY": "Chất lượng tổng thể",
    "PRODUCT#MATERIAL": "Chất liệu",
    "PRODUCT#COMFORT": "Cảm giác sử dụng",
    "PRODUCT#DESIGN": "Thiết kế & kiểu dáng",
    "PRODUCT#SAFETY": "Độ an toàn",
    "PRODUCT#FUNCTION": "Tính năng & công dụng",
    "PRODUCT#DURABILITY": "Độ bền",
    "PRODUCT#VALUE": "Giá trị so với giá tiền",
    "PRICE#AFFORDABILITY": "Mức giá",
    "PRICE#DISCOUNT": "Ưu đãi & khuyến mãi",
    "DELIVERY#SPEED": "Tốc độ giao hàng",
    "DELIVERY#PACKAGING": "Chất lượng đóng gói",
    "DELIVERY#ACCURACY": "Độ chính xác đơn hàng",
    "SELLER#SERVICE": "Chất lượng dịch vụ của shop",
    "SELLER#RESPONSIVENESS": "Mức độ phản hồi của shop",
    "SELLER#AUTHENTICITY": "Độ tin cậy / chính hãng",
}

ASPECT_ICONS = {
    "PRODUCT#QUALITY": "🧵", "PRODUCT#MATERIAL": "🪡", "PRODUCT#COMFORT": "😌",
    "PRODUCT#SIZE": "📐", "PRODUCT#DESIGN": "🎨", "PRODUCT#SAFETY": "🛡️",
    "PRODUCT#FUNCTION": "⚙️", "PRODUCT#DURABILITY": "💪", "PRODUCT#VALUE": "💰",
    "PRICE#AFFORDABILITY": "🏷️", "PRICE#DISCOUNT": "🎫",
    "DELIVERY#SPEED": "🚚", "DELIVERY#PACKAGING": "📦", "DELIVERY#ACCURACY": "✅",
    "SELLER#SERVICE": "🛒", "SELLER#RESPONSIVENESS": "💬", "SELLER#AUTHENTICITY": "🔐",
}

SENTIMENT_NAMES = ["none", "positive", "neutral", "negative"]
SENTIMENT_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
