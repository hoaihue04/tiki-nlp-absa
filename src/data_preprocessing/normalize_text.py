import pandas as pd
import regex as re
import emoji
import sys
import os

# FIX PATH
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, "../../"))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from src.utils.constants import CLEANED_REVIEWS_FILE, NORMALIZED_REVIEWS_FILE

# ============================================================
# NORMALIZATION DICTIONARY
# Phạm vi: Đánh giá sản phẩm mẹ & bé trên Tiki
# Bao gồm: Teencode · Viết tắt · Từ chuyên ngành · Cảm xúc
# ============================================================

# ────────────────────────────────────────────────────────────
# 1. CHUẨN HÓA DẤU & KÝ TỰ ĐẶC BIỆT
# ────────────────────────────────────────────────────────────
DIACRITIC_DICT = {
    'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
    'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
    'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
    'uả': 'ủa', 'ả': 'ả', 'ố': 'ố', 'u´': 'ố', 'ỗ': 'ỗ',
    'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ', 'ầ': 'ầ',
    'ỏ': 'ỏ', 'ề': 'ề', 'ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế',
    'ở': 'ở', 'ỉ': 'ỉ', 'ẻ': 'ẻ', 'àk': ' à ', 'aˋ': 'à',
    'iˋ': 'ì', 'ă´': 'ắ', 'ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ',
    'a´': 'á', 'ể': 'ể', '"': ' ',
}

# ────────────────────────────────────────────────────────────
# 2. SENTIMENT / TIẾNG ANH THÔNG DỤNG
# ────────────────────────────────────────────────────────────
SENTIMENT_DICT = {
    # OK variants
    'ô kêi': ' ok ', 'okie': ' ok ', 'o kê': ' ok ', 'okey': ' ok ',
    'ôkê': ' ok ', 'oki': ' ok ', 'oke': ' ok ', 'okay': ' ok ',
    'okê': ' ok ', 'ok': ' tốt ',

    # Cảm ơn
    'tks': ' cảm ơn ', 'thks': ' cảm ơn ', 'thanks': ' cảm ơn ',
    'ths': ' cảm ơn ', 'thank': ' cảm ơn ', 'thank you': ' cảm ơn ',
    'ty': ' cảm ơn ', 'tq': ' cảm ơn ',

    # Tích cực
    '⭐': ' star ', '*': ' star ', '🌟': ' star ', '🎉': ' tích cực ',
    'he he': ' tích cực ', 'hehe': ' tích cực ', 'hihi': ' tích cực ',
    'haha': ' tích cực ', 'hjhj': ' tích cực ', ':)': ' tích cực ',
    ':D': ' tích cực ', '^^': ' tích cực ', '<3': ' yêu thích ',
    '😍': ' yêu thích ', '😊': ' tích cực ', '🥰': ' yêu thích ',
    '👍': ' tốt ', '💯': ' rất tốt ', '🔥': ' rất tốt ',

    # Tiêu cực
    ':(' : ' tiêu cực ', '😢': ' tiêu cực ', '😡': ' tức giận ',
    '👎': ' tệ ', '😤': ' thất vọng ',
    'lol': ' tiêu cực ', 'cc': ' tiêu cực ', 'huhu': ' tiêu cực ',
    'hix': ' tiêu cực ', 'hic': ' tiêu cực ', 'híc': ' tiêu cực ',
    'ugh': ' tiêu cực ',

    # Chất lượng - tích cực
    'gud': ' tốt ', 'good': ' tốt ', 'gút': ' tốt ', 'gut': ' tốt ',
    'nice': ' tốt ', 'perfect': ' hoàn hảo ', 'excellent': ' xuất sắc ',
    'excelent': ' xuất sắc ', 'great': ' tuyệt vời ', 'amazing': ' tuyệt vời ',
    'awesome': ' tuyệt vời ', 'wonderful': ' tuyệt vời ',
    'superb': ' tuyệt vời ', 'love': ' yêu thích ', 'like': ' thích ',
    'wel done': ' tốt ', 'well done': ' tốt ',

    # Chất lượng - tiêu cực
    'bad': ' tệ ', 'sad': ' buồn ', 'poor': ' kém ', 'por': ' kém ',
    'terrible': ' rất tệ ', 'horrible': ' rất tệ ', 'awful': ' rất tệ ',
    'worst': ' tệ nhất ', 'disappoint': ' thất vọng ',
    'disappointed': ' thất vọng ', 'disappointing': ' thất vọng ',
    'waste': ' lãng phí ', 'sấu': ' xấu ', 'xau': ' xấu ',

    # Mô tả thêm
    'very': ' rất ', 'so': ' rất ', 'super': ' siêu ',
    'too': ' quá ', 'really': ' thực sự ', 'absolutely': ' hoàn toàn ',
    'beautiful': ' đẹp tuyệt vời ', 'cute': ' dễ thương ',
    'pretty': ' xinh xắn ', 'soft': ' mềm mại ', 'smooth': ' mịn màng ',
    'fresh': ' tươi mới ', 'clean': ' sạch sẽ ', 'safe': ' an toàn ',
    'quickly': ' nhanh ', 'quick': ' nhanh ', 'fast': ' nhanh ',
    'slow': ' chậm ', 'slowly': ' chậm ',
    'cheap': ' rẻ ', 'expensive': ' đắt ', 'affordable': ' hợp lý ',
    'delicious': ' ngon ', 'yummy': ' ngon ',
    'strong': ' chắc chắn ', 'sturdy': ' chắc chắn ',
    'durable': ' bền ', 'fragile': ' dễ vỡ ', 'light': ' nhẹ ',
    'heavy': ' nặng ', 'thick': ' dày ', 'thin': ' mỏng ',
    'comfortable': ' thoải mái ', 'comfy': ' thoải mái ',
    'convenient': ' tiện lợi ', 'easy': ' dễ dàng ',
    'difficult': ' khó khăn ', 'hard': ' khó ',
    'natural': ' tự nhiên ', 'organic': ' hữu cơ ',
    'genuine': ' chính hãng ', 'original': ' chính hãng ',
    'authentic': ' chính hãng ', 'fake': ' hàng giả ',
    'counterfeit': ' hàng giả ',
}

# ────────────────────────────────────────────────────────────
# 3. PHỦ ĐỊNH & TEENCODE CHUNG
# ────────────────────────────────────────────────────────────
TEEN_DICT = {
    # Phủ định
    'ko': ' không ', 'kg': ' không ', 'not': ' không ',
    'kh': ' không ', 'kô': ' không ', 'hok': ' không ',
    'k': ' không ', 'khong': ' không ', 'kp': ' không phải ',
    'ko phải': ' không phải ', 'chưa': ' chưa ',

    # Khẳng định / Phó từ
    'dc': ' được ', 'đc': ' được ', 'dk': ' được ', 'đk': ' được ', 'đx': ' được ',
    'r': ' rồi ', 'rr': ' rồi ', 'ròi': ' rồi ',
    'vs': ' với ', 'wa': ' quá ', 'wá': ' quá ', 'qá': ' quá ',
    'j': ' gì ', 'z': ' vậy ', 'v': ' vậy ', 'vz': ' vậy ',
    'nha': ' nhé ', 'nhá': ' nhé ', 'nhe': ' nhé ',
    'ạ': ' ạ ', 'ơi': ' ơi ',
    'thoy': ' thôi ', 'thui': ' thôi ',
    'cx': ' cũng ', 'cg': ' cũng ',
    'mk': ' mình ', 'mik': ' mình ', 'mh': ' mình ',
    'bn': ' bạn ', 'bạn ơi': ' bạn ',
    'ms': ' mới ', 'nx': ' nữa ',
    'bjo': ' bao giờ ', 'bh': ' bao giờ ',
    'nt': ' nhắn tin ', 'ib': ' nhắn tin ', 'inbox': ' nhắn tin ',
    'tl': ' trả lời ', 'trl': ' trả lời ', 'rep': ' trả lời ', 'reply': ' trả lời ',
    'trc': ' trước ', 'trc khi': ' trước khi ',
    'sau khi': ' sau khi ', 'saukhi': ' sau khi ',
    'vd': ' ví dụ ', 'vidu': ' ví dụ ',
    'đb': ' đặc biệt ', 'đặc biet': ' đặc biệt ',
    'ncl': ' nói chung là ', 'ngl': ' nói chung là ',
    'bth': ' bình thường ', 'bt': ' bình thường ',
    'hsd': ' hạn sử dụng ', 'date': ' hạn sử dụng ',
    'sd': ' sử dụng ', 'sài': ' xài ', 'xài': ' xài ',
    'time': ' thời gian ',
}

# ────────────────────────────────────────────────────────────
# 4. THƯƠNG MẠI ĐIỆN TỬ & DỊCH VỤ (TIKI)
# ────────────────────────────────────────────────────────────
ECOMMERCE_DICT = {
    # Sản phẩm / đơn hàng
    'sp': ' sản phẩm ', 'product': ' sản phẩm ', 'item': ' sản phẩm ',
    'hàg': ' hàng ', 'hang': ' hàng ', 'goods': ' hàng hóa ',
    'order': ' đặt hàng ', 'đơn': ' đơn hàng ', 'bill': ' đơn hàng ',
    'invoice': ' hóa đơn ', 'receipt': ' biên lai ',

    # Giao hàng
    'ship': ' giao hàng ', 'delivery': ' giao hàng ', 'síp': ' giao hàng ',
    'shipper': ' người giao hàng ', 'freeship': ' miễn phí giao hàng ',
    'free ship': ' miễn phí giao hàng ', 'freeshiping': ' miễn phí giao hàng ',
    'shipping': ' phí giao hàng ', 'vận chuyển': ' vận chuyển ',
    'vc': ' vận chuyển ', 'cod': ' thanh toán khi nhận hàng ',

    # Người bán / cửa hàng
    'shop': ' cửa hàng ', 'store': ' cửa hàng ', 'sop': ' cửa hàng ',
    'shopE': ' cửa hàng ', 'seller': ' người bán ', 'nv': ' nhân viên ',
    'cs': ' chăm sóc khách hàng ', 'cskh': ' chăm sóc khách hàng ',
    'admin': ' quản trị viên ', 'mod': ' quản lý ',

    # Đánh giá / phản hồi
    'fback': ' phản hồi ', 'fedback': ' phản hồi ', 'feedback': ' phản hồi ',
    'review': ' đánh giá ', 'rate': ' đánh giá ', 'rating': ' đánh giá ',
    'cmt': ' bình luận ', 'comment': ' bình luận ',

    # Chính hãng / hàng giả
    'auth': ' chính hãng ', 'aut': ' chính hãng ',
    'authentic': ' chính hãng ', 'genuine': ' chính hãng ',
    'fake': ' hàng giả ', 'nhái': ' hàng giả ', 'kém chất lượng': ' kém chất lượng ',

    # Giá cả / khuyến mãi
    'kc': ' khuyến mãi ', 'km': ' khuyến mãi ', 'promo': ' khuyến mãi ',
    'sale': ' giảm giá ', 'discount': ' giảm giá ', 'giamgia': ' giảm giá ',
    'gg': ' giảm giá ', 'deal': ' ưu đãi ', 'voucher': ' phiếu giảm giá ',
    'coupon': ' phiếu giảm giá ', 'flashsale': ' flash sale ',
    'flash sale': ' flash sale ', 'combo': ' combo ',

    # Đóng gói
    'pk': ' đóng gói ', 'pack': ' đóng gói ', 'packaging': ' đóng gói ',
    'box': ' hộp ', 'túi': ' túi ', 'bag': ' túi ',
    'carton': ' thùng carton ', 'bubble': ' túi bong bóng ',

    # Chất lượng chung
    'quality': ' chất lượng ', 'chất lg': ' chất lượng ',
    'chất lg': ' chất lượng ', 'chat': ' chất ',
    'cl': ' chất lượng ',

    # Nền tảng
    'tiki': ' tiki ', 'shopee': ' shopee ', 'lazada': ' lazada ',
    'sendo': ' sendo ', 'fb': ' facebook ', 'face': ' facebook ',

    # Size
    'sz': ' cỡ ', 'size': ' cỡ ', 'form': ' kiểu dáng ',
    'fit': ' vừa vặn ', 'loose': ' rộng ', 'tight': ' chật ',
}

# ────────────────────────────────────────────────────────────
# 5. SẢN PHẨM MẸ & BÉ – CHUYÊN NGÀNH
# ────────────────────────────────────────────────────────────

# 5a. Tã / Bỉm
DIAPER_DICT = {
    'bim': ' bỉm ', 'ta': ' tã ', 'tả': ' tã ', 'tã bỉm': ' tã bỉm ',
    'diaper': ' tã ', 'nappy': ' tã ', 'nappies': ' tã ',
    'pull up': ' tã quần ', 'pullup': ' tã quần ', 'tã quần': ' tã quần ',
    'tã dán': ' tã dán ', 'tã miếng': ' tã miếng ',
    'tã sơ sinh': ' tã sơ sinh ', 'tã trẻ em': ' tã trẻ em ',

    # Size tã
    'nb': ' newborn ', 'size nb': ' sơ sinh ', 'newborn': ' sơ sinh ',
    'size s': ' size s ', 'size m': ' size m ', 'size l': ' size l ',
    'size xl': ' size xl ', 'size xxl': ' size xxl ',
    'ssm': ' size m ', 'ssl': ' size l ',

    # Thương hiệu tã phổ biến
    'pampers': ' pampers ', 'huggies': ' huggies ',
    'bobby': ' bobby ', 'bim bobby': ' tã bobby ',
    'merries': ' merries ', 'goo.n': ' goon ', 'goon': ' goon ',
    'moony': ' moony ', 'mamypoko': ' mamypoko ',
    'nakies': ' nakies ', 'gnappies': ' gnappies ',
    'bimbo': ' bimbo ', 'picolin': ' picolin ',

    # Đặc tính tã
    'thấm hút': ' thấm hút ', 'tham hut': ' thấm hút ',
    'không rò rỉ': ' không rò rỉ ', 'ro ri': ' rò rỉ ', 'rò rỉ': ' rò rỉ ',
    'mềm mại': ' mềm mại ', 'mem mai': ' mềm mại ',
    'thoáng khí': ' thoáng khí ', 'thoang khi': ' thoáng khí ',
    'không hăm': ' không hăm da ', 'hăm': ' hăm da ', 'ham': ' hăm da ',
    'hăm da': ' hăm da ', 'han da': ' hăm da ',
    'không gây dị ứng': ' không gây dị ứng ',
    'dị ứng': ' dị ứng ', 'di ung': ' dị ứng ',
    'kích ứng': ' kích ứng da ',
}

# 5b. Sữa công thức / Dinh dưỡng
FORMULA_DICT = {
    'sua': ' sữa ', 'milk': ' sữa ',
    'sữa ct': ' sữa công thức ', 'scf': ' sữa công thức ',
    'sữa công thức': ' sữa công thức ', 'sua cong thuc': ' sữa công thức ',
    'sữa bột': ' sữa bột ', 'sua bot': ' sữa bột ',
    'sữa nước': ' sữa nước ', 'sữa tươi': ' sữa tươi ',
    'sữa mẹ': ' sữa mẹ ', 'sữa mẹ em bé': ' sữa mẹ ',
    'bú mẹ': ' bú mẹ ', 'cho bú': ' cho con bú ',
    'hút sữa': ' hút sữa ', 'máy hút sữa': ' máy hút sữa ',

    # Thương hiệu sữa phổ biến
    'nan': ' nan ', 'similac': ' similac ', 'enfamil': ' enfamil ',
    'enfagrow': ' enfagrow ', 'enfa': ' enfamil ',
    'aptamil': ' aptamil ', 'dumex': ' dumex ',
    'friso': ' friso ', 'frisolac': ' frisolac ',
    'nutifood': ' nutifood ', 'nuti': ' nutifood ',
    'vinamilk': ' vinamilk ', 'dutch lady': ' dutch lady ',
    'meiji': ' meiji ', 'wakodo': ' wakodo ',
    'glico': ' glico ', 'morinaga': ' morinaga ',
    's26': ' s26 ', 'nestle': ' nestle ', 'nestlé': ' nestle ',
    'nutrilon': ' nutrilon ', 'hero baby': ' hero baby ',
    'hipp': ' hipp ', 'holle': ' holle ', 'humana': ' humana ',
    'nutrilac': ' nutrilac ', 'dielac': ' dielac ',
    'grow': ' grow ', 'pediasure': ' pediasure ',

    # Loại sữa / giai đoạn
    'stage 1': ' giai đoạn 1 ', 'stage 2': ' giai đoạn 2 ',
    'stage 3': ' giai đoạn 3 ',
    'ct1': ' công thức 1 ', 'ct2': ' công thức 2 ', 'ct3': ' công thức 3 ',
    'số 1': ' số 1 ', 'số 2': ' số 2 ', 'số 3': ' số 3 ',
    'lon': ' lon sữa ', 'hộp': ' hộp sữa ',
    'túi': ' túi sữa ', 'thanh': ' thanh sữa ',

    # Dinh dưỡng
    'protein': ' protein ', 'dha': ' dha ', 'ara': ' ara ',
    'iq': ' phát triển trí não ', 'omega': ' omega ',
    'canxi': ' canxi ', 'calcium': ' canxi ',
    'vitamin': ' vitamin ', 'khoáng chất': ' khoáng chất ',
    'iron': ' sắt ', 'sắt': ' sắt ', 'zinc': ' kẽm ', 'kẽm': ' kẽm ',
    'prebiotics': ' prebiotics ', 'probiotics': ' probiotics ',
    'lactose': ' lactose ', 'lactose free': ' không lactose ',
    'không lactose': ' không lactose ',

    # Đồ ăn dặm
    'an dam': ' ăn dặm ', 'ăn dặm': ' ăn dặm ', 'dam': ' ăn dặm ',
    'bot an dam': ' bột ăn dặm ', 'bột ăn dặm': ' bột ăn dặm ',
    'cháo': ' cháo ', 'nước cháo': ' nước cháo ',
    'rau củ': ' rau củ ', 'trái cây': ' trái cây ',
    'puree': ' cháo xay ', 'pured': ' cháo xay ',
    'snack': ' đồ ăn vặt trẻ em ', 'bánh': ' bánh ',
    'bánh ăn dặm': ' bánh ăn dặm ',

    # Bình sữa / Dụng cụ ăn uống
    'binh sua': ' bình sữa ', 'bình sữa': ' bình sữa ', 'bottle': ' bình sữa ',
    'núm ti': ' núm ti ', 'num ti': ' núm ti ', 'nipple': ' núm ti ',
    'teat': ' núm ti ', 'núm vú': ' núm vú giả ',
    'ti giả': ' núm vú giả ', 'pacifier': ' núm vú giả ',
    'ty giả': ' núm vú giả ', 'soother': ' núm vú giả ',
    'bình hâm sữa': ' bình hâm sữa ', 'máy hâm sữa': ' máy hâm sữa ',
    'hâm sữa': ' hâm sữa ', 'ham sua': ' hâm sữa ',
    'máy tiệt trùng': ' máy tiệt trùng ', 'tiet trung': ' tiệt trùng ',
    'sterilizer': ' máy tiệt trùng ',
    'thìa': ' thìa ', 'muỗng': ' muỗng ', 'spoon': ' muỗng ',
    'tô': ' tô ', 'bowl': ' tô ', 'đĩa': ' đĩa ',
    'cốc': ' cốc ', 'cup': ' cốc ', 'sippy cup': ' cốc tập uống ',
    'straw cup': ' cốc ống hút ', 'straw': ' ống hút ',
    'yếm': ' yếm ', 'bib': ' yếm ', 'khăn yếm': ' yếm ',
}

# 5c. Quần áo trẻ em
CLOTHING_DICT = {
    'qa': ' quần áo ', 'ao': ' áo ', 'quan': ' quần ',
    'bodysuit': ' bodysuit ', 'romper': ' romper ',
    'onesie': ' onesie ', 'jumper': ' đồ liền thân ',
    'outfit': ' bộ đồ ', 'set đồ': ' bộ đồ ',
    'pyjama': ' đồ ngủ ', 'pajama': ' đồ ngủ ',
    'bộ ngủ': ' đồ ngủ ', 'đồ ngủ': ' đồ ngủ ',
    'áo thun': ' áo thun ', 'áo sơ mi': ' áo sơ mi ',
    'áo khoác': ' áo khoác ', 'áo len': ' áo len ',
    'váy': ' váy ', 'dress': ' váy đầm ',
    'quần short': ' quần short ', 'short': ' quần ngắn ',
    'legging': ' quần legging ', 'quần dài': ' quần dài ',

    # Chất liệu quần áo
    'cotton': ' cotton ', 'coton': ' cotton ', 'kaki': ' kaki ',
    'vải cotton': ' vải cotton ', 'vải lụa': ' vải lụa ',
    'vải muslin': ' vải muslin ', 'muslin': ' vải muslin ',
    'vải bamboo': ' vải tre ', 'bamboo': ' vải tre ',
    'sợi tre': ' vải tre ', 'vải jersey': ' vải jersey ',
    'polyester': ' polyester ', 'modal': ' vải modal ',
    'linen': ' vải linen ', 'fleece': ' vải lông cừu ',

    # Đặc tính vải
    'thoáng mát': ' thoáng mát ', 'thoang mat': ' thoáng mát ',
    'mềm': ' mềm ', 'mem': ' mềm ',
    'co giãn': ' co giãn ', 'co dan': ' co giãn ',
    'thấm mồ hôi': ' thấm mồ hôi ', 'tham mo hoi': ' thấm mồ hôi ',
    'kháng khuẩn': ' kháng khuẩn ', 'khang khuan': ' kháng khuẩn ',
    'an toàn': ' an toàn ', 'an toan': ' an toàn ',
    'không phai màu': ' không phai màu ', 'phai màu': ' phai màu ',
    'bền màu': ' bền màu ',

    # Size quần áo trẻ em
    '0-3m': ' 0 đến 3 tháng ', '3-6m': ' 3 đến 6 tháng ',
    '6-9m': ' 6 đến 9 tháng ', '9-12m': ' 9 đến 12 tháng ',
    '12-18m': ' 12 đến 18 tháng ', '18-24m': ' 18 đến 24 tháng ',
    '2t': ' 2 tuổi ', '3t': ' 3 tuổi ', '4t': ' 4 tuổi ',
    '5t': ' 5 tuổi ', '1y': ' 1 tuổi ', '2y': ' 2 tuổi ',
    'newborn': ' sơ sinh ', 'nb': ' sơ sinh ',
}

# 5d. Chăm sóc da & Vệ sinh
SKINCARE_DICT = {
    # Khăn ướt
    'khan uot': ' khăn ướt ', 'khăn ướt': ' khăn ướt ',
    'wet wipe': ' khăn ướt ', 'wipe': ' khăn ướt ',
    'khăn giấy': ' khăn giấy ', 'tissue': ' khăn giấy ',
    'khăn bông': ' khăn bông ',

    # Sản phẩm tắm gội
    'sua tam': ' sữa tắm ', 'sữa tắm': ' sữa tắm ',
    'dầu gội': ' dầu gội ', 'dau goi': ' dầu gội ', 'shampoo': ' dầu gội ',
    'tắm gội': ' tắm gội ', 'tam goi': ' tắm gội ',
    'gel tắm': ' gel tắm ', 'shower gel': ' gel tắm ',
    'xà phòng': ' xà phòng ', 'soap': ' xà phòng ',
    'bồn tắm': ' bồn tắm ', 'chậu tắm': ' chậu tắm ',
    'ghế tắm': ' ghế tắm ', 'lưới tắm': ' lưới tắm ',

    # Dưỡng da
    'kem dưỡng': ' kem dưỡng da ', 'kem duong': ' kem dưỡng da ',
    'lotion': ' lotion dưỡng da ', 'moisturizer': ' kem dưỡng ẩm ',
    'kem chống hăm': ' kem chống hăm ', 'kem chong ham': ' kem chống hăm ',
    'kem hăm': ' kem chống hăm ', 'diaper cream': ' kem chống hăm ',
    'vaseline': ' vaseline ', 'dầu olive': ' dầu olive ',
    'dầu dừa': ' dầu dừa ', 'coconut oil': ' dầu dừa ',
    'baby oil': ' dầu dưỡng trẻ em ',
    'phấn': ' phấn rôm ', 'powder': ' phấn rôm ',
    'sunscreen': ' kem chống nắng ', 'kem chống nắng': ' kem chống nắng ',

    # An toàn da
    'paraben free': ' không paraben ', 'không paraben': ' không paraben ',
    'sulfate free': ' không sulfate ', 'không sulfate': ' không sulfate ',
    'fragrance free': ' không mùi ', 'không mùi': ' không mùi ',
    'hypoallergenic': ' không gây dị ứng ', 'ph balance': ' cân bằng ph ',
    'pediatrician': ' được bác sĩ kiểm định ',
    'dermatologist': ' được kiểm định da liễu ',
    'clinically tested': ' được kiểm nghiệm lâm sàng ',
}

# 5e. Đồ chơi & Phát triển
TOYS_DICT = {
    'do choi': ' đồ chơi ', 'đồ chơi': ' đồ chơi ', 'toy': ' đồ chơi ',
    'toys': ' đồ chơi ', 'plaything': ' đồ chơi ',
    'educational toy': ' đồ chơi giáo dục ',
    'learning toy': ' đồ chơi học tập ',
    'stem': ' stem ', 'stem toy': ' đồ chơi stem ',

    # Loại đồ chơi
    'rattles': ' đồ chơi lúc lắc ', 'rattle': ' đồ chơi lúc lắc ',
    'luc lac': ' lúc lắc ', 'mobile': ' đồ chơi treo nôi ',
    'xúc xắc': ' xúc xắc ', 'xuc xac': ' xúc xắc ',
    'khối gỗ': ' khối gỗ ', 'khoi go': ' khối gỗ ', 'block': ' khối xếp hình ',
    'lego': ' lego ', 'xếp hình': ' xếp hình ',
    'puzzle': ' xếp hình ghép mảnh ',
    'stuffed animal': ' thú nhồi bông ', 'plush': ' thú nhồi bông ',
    'bông': ' thú nhồi bông ', 'gấu bông': ' gấu bông ',
    'búp bê': ' búp bê ', 'doll': ' búp bê ',
    'xe đồ chơi': ' xe đồ chơi ', 'ô tô đồ chơi': ' ô tô đồ chơi ',
    'nhạc cụ': ' nhạc cụ đồ chơi ', 'piano đồ chơi': ' piano đồ chơi ',
    'sách': ' sách ', 'book': ' sách ',
    'flashcard': ' flashcard ', 'flash card': ' flashcard ',
    'bảng vẽ': ' bảng vẽ ', 'màu vẽ': ' màu vẽ ', 'crayon': ' bút màu ',
    'sticker': ' nhãn dán ', 'nhãn dán': ' nhãn dán ',

    # Phát triển
    'phát triển': ' phát triển ', 'phat trien': ' phát triển ',
    'kích thích': ' kích thích ', 'kich thich': ' kích thích ',
    'sáng tạo': ' sáng tạo ', 'sang tao': ' sáng tạo ',
    'ngôn ngữ': ' ngôn ngữ ', 'ngon ngu': ' ngôn ngữ ',
    'vận động': ' vận động ', 'van dong': ' vận động ',
    'giác quan': ' giác quan ', 'giac quan': ' giác quan ',
    'trí tuệ': ' trí tuệ ', 'tri tue': ' trí tuệ ',
}

# 5f. Đồ dùng phòng bé / Nôi cũi / An toàn
NURSERY_DICT = {
    # Ngủ
    'noi': ' nôi ', 'cũi': ' cũi ', 'cui': ' cũi ',
    'bassinet': ' nôi sơ sinh ', 'crib': ' cũi ', 'cradle': ' nôi ',
    'cot': ' cũi ', 'playpen': ' cũi quây ',
    'mattress': ' đệm ', 'nệm': ' đệm ', 'gối': ' gối ',
    'gối ôm': ' gối ôm ', 'goi om': ' gối ôm ',
    'chăn': ' chăn ', 'mền': ' chăn ', 'blanket': ' chăn ',
    'swaddle': ' quấn ủ trẻ ', 'quấn ủ': ' quấn ủ trẻ ',
    'sleeping bag': ' túi ngủ em bé ', 'túi ngủ': ' túi ngủ em bé ',

    # Di chuyển
    'xe đẩy': ' xe đẩy em bé ', 'xe day': ' xe đẩy em bé ',
    'stroller': ' xe đẩy em bé ', 'pram': ' xe đẩy em bé ',
    'buggy': ' xe đẩy nhỏ ', 'jogger': ' xe đẩy chạy bộ ',
    'địu': ' địu em bé ', 'diu': ' địu em bé ',
    'baby carrier': ' địu em bé ', 'wrap': ' địu vải ',
    'sling': ' địu hông ', 'ergobaby': ' địu ergobaby ',
    'ghế ô tô': ' ghế ô tô em bé ', 'ghe o to': ' ghế ô tô em bé ',
    'car seat': ' ghế ô tô em bé ', 'infant seat': ' ghế ô tô sơ sinh ',
    'booster seat': ' ghế nâng ngồi ăn ',
    'highchair': ' ghế ngồi ăn ', 'high chair': ' ghế ngồi ăn ',
    'ghế ăn': ' ghế ăn dặm ', 'ghe an': ' ghế ăn dặm ',

    # Phụ kiện phòng bé
    'máy đo nhiệt độ': ' nhiệt kế ', 'nhiet ke': ' nhiệt kế ',
    'thermometer': ' nhiệt kế ', 'nhiệt kế': ' nhiệt kế ',
    'máy phun sương': ' máy phun sương ', 'humidifier': ' máy tạo ẩm ',
    'máy lọc không khí': ' máy lọc không khí ',
    'monitor': ' máy theo dõi em bé ', 'baby monitor': ' máy theo dõi em bé ',
    'đèn ngủ': ' đèn ngủ ', 'nightlight': ' đèn ngủ ',

    # An toàn
    'an toàn': ' an toàn ', 'safety': ' an toàn ',
    'chứng nhận': ' chứng nhận ', 'certified': ' đạt chuẩn ',
    'fda': ' fda ', 'ce': ' đạt chuẩn ce ', 'astm': ' astm ',
    'tiêu chuẩn': ' tiêu chuẩn ', 'tieu chuan': ' tiêu chuẩn ',
    'bpa free': ' không bpa ', 'không bpa': ' không bpa ',
    'phthalate free': ' không phthalate ',
    'không độc hại': ' không độc hại ', 'non toxic': ' không độc hại ',
}

# 5g. Sức khỏe & Y tế trẻ em
HEALTH_DICT = {
    # Thuốc / Chăm sóc y tế
    'siro': ' siro ', 'nhỏ giọt': ' nhỏ giọt ',
    'vitamin d': ' vitamin d ', 'vit d': ' vitamin d ',
    'vitamin c': ' vitamin c ', 'vit c': ' vitamin c ',
    'fish oil': ' dầu cá ', 'dầu cá': ' dầu cá ',
    'omega 3': ' omega 3 ', 'omega3': ' omega 3 ',
    'sắt': ' sắt ', 'iron': ' sắt ', 'kẽm': ' kẽm ',
    'canxi': ' canxi ', 'men vi sinh': ' men vi sinh ',
    'probiotic': ' men vi sinh ', 'prebiotic': ' prebiotic ',
    'vaccine': ' vaccine ', 'tiêm chủng': ' tiêm chủng ',

    # Tình trạng sức khỏe
    'sốt': ' sốt ', 'sot': ' sốt ', 'fever': ' sốt ',
    'ho': ' ho ', 'cough': ' ho ', 'sổ mũi': ' sổ mũi ',
    'so mui': ' sổ mũi ', 'runny nose': ' sổ mũi ',
    'tiêu chảy': ' tiêu chảy ', 'tieu chay': ' tiêu chảy ',
    'táo bón': ' táo bón ', 'tao bon': ' táo bón ',
    'đau bụng': ' đau bụng ', 'dau bung': ' đau bụng ',
    'đầy hơi': ' đầy hơi ', 'day hoi': ' đầy hơi ',
    'nôn trớ': ' nôn trớ ', 'non tro': ' nôn trớ ', 'spit up': ' nôn trớ ',
    'colicky': ' đau bụng colic ', 'colic': ' đau bụng colic ',
    'mọc răng': ' mọc răng ', 'moc rang': ' mọc răng ', 'teething': ' mọc răng ',
    'hăm': ' hăm da ', 'ham': ' hăm da ', 'diaper rash': ' hăm tã ',
    'dị ứng': ' dị ứng ', 'di ung': ' dị ứng ', 'allergy': ' dị ứng ',
    'chàm': ' chàm da ', 'eczema': ' chàm da ',
    'bé khóc': ' bé khóc ', 'khó ngủ': ' khó ngủ ',

    # Thiết bị y tế
    'máy hút mũi': ' máy hút mũi ', 'hut mui': ' hút mũi ',
    'nasal aspirator': ' máy hút mũi ',
    'kẹp rốn': ' kẹp rốn ', 'keo ron': ' kẹp rốn ',
    'bộ vệ sinh': ' bộ vệ sinh ', 'grooming kit': ' bộ vệ sinh ',
    'kéo cắt móng': ' kéo cắt móng tay ',
    'nail clipper': ' kéo cắt móng tay ',
}

# 5h. Địa điểm / Người dùng
DEMOGRAPHY_DICT = {
    'hcm': ' hồ chí minh ', 'tphcm': ' hồ chí minh ', 'sài gòn': ' hồ chí minh ',
    'hn': ' hà nội ', 'hanoi': ' hà nội ',
    'đn': ' đà nẵng ', 'da nang': ' đà nẵng ',
    'cần thơ': ' cần thơ ', 'can tho': ' cần thơ ',
    'binh duong': ' bình dương ', 'đồng nai': ' đồng nai ',
    'cty': ' công ty ', 'vn': ' việt nam ',

    # Đối tượng
    'be': ' bé ', 'bb': ' bé ', 'baby': ' em bé ',
    'con': ' con ', 'bé yêu': ' bé yêu ',
    'sơ sinh': ' sơ sinh ', 'so sinh': ' sơ sinh ',
    'trẻ sơ sinh': ' trẻ sơ sinh ', 'newborn': ' trẻ sơ sinh ',
    'trẻ em': ' trẻ em ', 'tre em': ' trẻ em ',
    'infant': ' trẻ sơ sinh ', 'toddler': ' trẻ chập chững ',
    'bé gái': ' bé gái ', 'bé trai': ' bé trai ',
    'mẹ': ' mẹ ', 'ba': ' ba ', 'bố': ' bố ', 'baba': ' bố ',
    'mẹ bầu': ' mẹ bầu ', 'bầu': ' mang thai ', 'thai kỳ': ' thai kỳ ',
    'sau sinh': ' sau sinh ', 'postpartum': ' sau sinh ',
}

# ────────────────────────────────────────────────────────────
# 6. GỘP TẤT CẢ VÀO MỘT DICT CHÍNH
#    Thứ tự quan trọng: dict sau ghi đè dict trước nếu trùng key
# ────────────────────────────────────────────────────────────
NORMALIZATION_DICT = {
    **DIACRITIC_DICT,
    **SENTIMENT_DICT,
    **TEEN_DICT,
    **ECOMMERCE_DICT,
    **DIAPER_DICT,
    **FORMULA_DICT,
    **CLOTHING_DICT,
    **SKINCARE_DICT,
    **TOYS_DICT,
    **NURSERY_DICT,
    **HEALTH_DICT,
    **DEMOGRAPHY_DICT,
}

# Dict viết tắt riêng (ưu tiên khớp toàn từ)
ABBREVIATION_DICT = {
    'hcm': ' hồ chí minh ',
    'tphcm': ' hồ chí minh ',
    'hn': ' hà nội ',
    'cty': ' công ty ',
    'sp': ' sản phẩm ',
    'nv': ' nhân viên ',
    'vc': ' vận chuyển ',
    'cs': ' chăm sóc khách hàng ',
    'cskh': ' chăm sóc khách hàng ',
    'kh': ' khách hàng ',
    'kq': ' kết quả ',
    'tg': ' thời gian ',
    'hsd': ' hạn sử dụng ',
    'sl': ' số lượng ',
    'cl': ' chất lượng ',
    'gd': ' giao dịch ',
    'gdt': ' giá đã tặng ',
    'kc': ' khuyến mãi ',
    'pk': ' phụ kiện ',
    'bb': ' bé ',
    'scf': ' sữa công thức ',
    'bps': ' bình sữa ',
}

# ========================
# LOAD DATA
# ========================
def load_data():
    try:
        df = pd.read_csv(CLEANED_REVIEWS_FILE)
        print(f"Loaded {len(df)} reviews")
        return df
    except Exception as e:
        print("Error loading file:", e)
        return pd.DataFrame()


# ========================
# NORMALIZE TEXT
# ========================
def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 2. Remove URL
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 3. Remove "xem thêm / thu gọn"
    text = re.sub(r'\b(xem thêm|thu gọn)\b', ' ', text)

    # 4. Replace emoji → text label (language='vi' không được hỗ trợ, dùng 'en')
    text = emoji.demojize(text, language='en')

    # 5. Chuẩn hóa số + đơn vị đo (tháng tuổi, cân nặng, ml, gram...)
    text = re.sub(r'(\d+)\s*(tháng)', r'\1 tháng', text)
    text = re.sub(r'(\d+)\s*(tuổi|t)', r'\1 tuổi', text)
    text = re.sub(r'(\d+)\s*(kg|gram|g)\b', r'\1 \2', text)
    text = re.sub(r'(\d+)\s*(ml|l)\b', r'\1 \2', text)
    text = re.sub(r'(\d+)\s*(sao|star|⭐)', r'\1 sao', text)

    # 6. Chuẩn hóa viết tắt & teencode (khớp toàn từ ưu tiên)
    words = text.split()
    new_words = []
    for w in words:
        if w in ABBREVIATION_DICT:
            new_words.append(ABBREVIATION_DICT[w])
        elif w in NORMALIZATION_DICT:
            new_words.append(NORMALIZATION_DICT[w])
        else:
            new_words.append(w)
    text = " ".join(new_words)

    # 7. Chuẩn hóa cụm từ (multi-word expressions) — sau khi join lại
    for phrase, replacement in NORMALIZATION_DICT.items():
        if ' ' in phrase:            # chỉ xử lý cụm nhiều từ
            text = text.replace(phrase, replacement)

    # 8. Remove repeated characters (quáaaaa → quáa)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)

    # 9. Remove special characters (giữ chữ, số, khoảng trắng)
    text = re.sub(r'[^\p{L}\p{N}\s]', ' ', text)

    # 10. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ========================
# NORMALIZE DATA
# ========================
def normalize_data(df):
    if df.empty:
        return df

    print("\n--- NORMALIZING TEXT ---")
    df = df.copy()

    # Chọn cột text đúng pipeline
    if 'cleaned_content' in df.columns:
        df['normalized_text'] = df['cleaned_content'].apply(normalize_text)
    elif 'content' in df.columns:
        df['normalized_text'] = df['content'].apply(normalize_text)
    else:
        print("❌ Không tìm thấy cột text phù hợp")
        return df

    # Loại bỏ bản ghi rỗng sau normalize
    df = df[df['normalized_text'].str.len() > 0]
    print(f"Remaining after normalize: {len(df)}")

    # ────────────────────────────────────────
    # LABEL (5 CLASS)
    # ────────────────────────────────────────
    if 'sentiment_label' in df.columns:
        label_map = {
            "very_negative": 0,
            "negative":      1,
            "neutral":       2,
            "positive":      3,
            "very_positive": 4,
        }
        df['label'] = df['sentiment_label'].map(label_map)

    return df


# ========================
# SAVE
# ========================
def save_data(df):
    os.makedirs(os.path.dirname(NORMALIZED_REVIEWS_FILE), exist_ok=True)
    df.to_csv(NORMALIZED_REVIEWS_FILE, index=False)
    print(f"✅ Saved normalized data → {NORMALIZED_REVIEWS_FILE}")


# ========================
# MAIN
# ========================
def main():
    df = load_data()
    if df.empty:
        return
    df = normalize_data(df)
    save_data(df)
    print("\n🎉 NORMALIZATION DONE!")


if __name__ == "__main__":
    main()