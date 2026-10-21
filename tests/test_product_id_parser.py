from __future__ import annotations

from app.web.product_parser import extract_product_id_from_url


def test_extract_product_id_from_tiki_product_url() -> None:
    url = "https://tiki.vn/sua-tam-em-be-p123456789.html?spid=987654321"

    assert extract_product_id_from_url(url) == "123456789"


def test_extract_product_id_from_query_fallback() -> None:
    url = "https://tiki.vn/product?product_id=24680"

    assert extract_product_id_from_url(url) == "24680"


def test_extract_product_id_returns_none_without_digits() -> None:
    assert extract_product_id_from_url("https://tiki.vn/san-pham.html") is None
