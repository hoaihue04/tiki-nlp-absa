

import json
import logging
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple
import httpx

from config import get_key_manager, KeySlot, PROVIDERS
from constants import ASPECT_CATEGORIES, MAX_REVIEW_LENGTH

logger = logging.getLogger(__name__)


# ── Tuning ────────────────────────────────────────────────────────────
_ENSEMBLE_RAW     = os.getenv("ENSEMBLE_MODE", "false").lower()
ENSEMBLE_MODE     = _ENSEMBLE_RAW not in ("false", "0", "")
ENSEMBLE_N        = int(_ENSEMBLE_RAW) if _ENSEMBLE_RAW in ("2", "3") else 2
MAX_RETRIES       = 6        # tăng lên để có đủ cơ hội sau khi chờ
REQUEST_TIMEOUT   = 50
TOKENS_PER_REVIEW = 380
MAX_TOKENS_CAP    = 8_000
MAX_WAIT_KEY      = 65.0     # chờ tối đa 65s (1 sliding window) cho key

_N_WORKERS = max(int(os.getenv("N_WORKERS", "0")) or 12, 1)
_PROVIDER_ORDER = ["groq", "gemini", "openai"]


# ── Noise filter ──────────────────────────────────────────────────────
_NOISE_RE = [
    re.compile(r"^[.\s!?*\-_👍❤️✅⭐🌟💯]{0,5}$"),
    re.compile(r"^(.)\1{5,}$"),
    re.compile(r"^\d+$"),
]

def _is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) < 4:
        return True
    return any(p.match(t) for p in _NOISE_RE)


# ── System prompt ─────────────────────────────────────────────────────
_SYSTEM = """Bạn là chuyên gia annotation ASQP cho reviews sản phẩm mẹ & bé trên Tiki (tiếng Việt).

NHIỆM VỤ: Trích xuất tất cả ASQP quadruples từ mỗi review.

CATEGORIES HỢP LỆ — chỉ dùng đúng các giá trị sau:
PRODUCT#QUALITY, PRODUCT#MATERIAL, PRODUCT#COMFORT,
PRODUCT#SIZE, PRODUCT#DESIGN, PRODUCT#SAFETY,
PRODUCT#FUNCTION, PRODUCT#DURABILITY, PRODUCT#VALUE,
PRICE#AFFORDABILITY, PRICE#DISCOUNT,
DELIVERY#SPEED, DELIVERY#PACKAGING, DELIVERY#ACCURACY,
SELLER#SERVICE, SELLER#RESPONSIVENESS, SELLER#AUTHENTICITY

QUY TẮC:
1. aspect_term  : cụm từ xuất hiện trong text, hoặc "NULL" nếu implicit
2. opinion_term : cụm từ thể hiện ý kiến, hoặc "NULL" nếu implicit
3. sentiment    : "positive" | "negative" | "neutral"
4. IMPLICIT: "tốt lắm"/"siêu thích"/"ổn"/"okk"/"hài lòng" →
   {aspect_term:"NULL", aspect_category:"PRODUCT#QUALITY", opinion_term:<từ đó>, sentiment:"positive"}
5. Mỗi review có thể có NHIỀU quadruples — đừng bỏ sót
6. Review rất ngắn (1–4 từ) vẫn phải extract nếu có sentiment rõ ràng

OUTPUT: JSON thuần — KHÔNG markdown, KHÔNG giải thích.
{"results":{"<review_id>":[{"aspect_term":"...","aspect_category":"...","opinion_term":"...","sentiment":"..."}]}}"""


def _user_prompt(batch: List[dict]) -> str:
    parts = ["Annotate các reviews sau:\n"]
    for item in batch:
        text = item["text"][:MAX_REVIEW_LENGTH]
        parts.append(f'ID: {item["review_id"]}\nText: "{text}"\n')
    parts.append('\nOutput JSON: {"results":{"<id>":[quadruples...]}}')
    return "\n".join(parts)


# ── HTTP client ───────────────────────────────────────────────────────
_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()

def _http() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    timeout=REQUEST_TIMEOUT,
                    limits=httpx.Limits(
                        max_connections=64,
                        max_keepalive_connections=32,
                    ),
                )
    return _client


# ── Raw API call ──────────────────────────────────────────────────────
def _call(slot: KeySlot, cfg: dict, messages: list, max_tokens: int) -> str:
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {slot.key}",
    }
    body = {
        "model":       cfg["model"],
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.05,
    }
    resp = _http().post(cfg["url"], json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Parse + validate ──────────────────────────────────────────────────
def _parse(raw: str, ids: List[str]) -> Dict[str, List[dict]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

    diff = raw.count("{") - raw.count("}")
    if 0 < diff <= 5:
        raw += "}" * diff
        logger.debug(f"🔧 Auto-fixed truncation: +{diff} '}}'")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"❌ JSON parse fail: {e} | preview: {raw[:120]!r}")
        return {}

    out: Dict[str, List[dict]] = {}
    for rid in ids:
        raw_quads = data.get("results", {}).get(str(rid), [])
        valid = []
        for q in raw_quads:
            if not isinstance(q, dict):
                continue
            cat  = q.get("aspect_category", "")
            sent = q.get("sentiment", "").lower()
            if cat not in ASPECT_CATEGORIES:
                logger.debug(f"⬛ Invalid category '{cat}' → drop")
                continue
            if sent not in ("positive", "negative", "neutral"):
                continue
            valid.append({
                "aspect_term":     str(q.get("aspect_term",  "NULL")),
                "aspect_category": cat,
                "opinion_term":    str(q.get("opinion_term", "NULL")),
                "sentiment":       sent,
            })
        out[rid] = valid
    return out


# ── Key availability helpers ──────────────────────────────────────────

def _next_available_in(provider: Optional[str] = None) -> float:
    """
    Trả về số giây cần chờ cho đến khi có ít nhất 1 key available.
    0.0 nếu đã có key sẵn.
    """
    mgr = get_key_manager()
    slots = [
        s for s in mgr._slots
        if provider is None or s.provider == provider
    ]
    if not slots:
        return MAX_WAIT_KEY

    # Key đã available → không cần chờ
    if any(s.is_available for s in slots):
        return 0.0

    # Tính thời gian hết cooldown gần nhất
    now = time.monotonic()
    earliest_free = min(
        s._cooldown_until if s._cooldown_until > now else now
        for s in slots
    )
    # Cộng thêm thời gian window sliding (60s) nếu key đang bị rate limit
    # nhưng không có cooldown riêng
    wait = earliest_free - now
    return max(0.1, min(wait + 0.5, MAX_WAIT_KEY))


def _wait_for_key(provider: Optional[str] = None, label: str = "") -> bool:
    """
    Chờ cho đến khi có key available (hoặc timeout MAX_WAIT_KEY).
    Trả về True nếu có key, False nếu timeout.
    """
    waited = 0.0
    while waited < MAX_WAIT_KEY:
        wait = _next_available_in(provider)
        if wait <= 0.0:
            return True
        # Giới hạn mỗi lần sleep tối đa 2s để responsive hơn
        sleep = min(wait, 2.0)
        logger.info(
            f"⏳ {label}Chờ key {provider or 'any'} ({wait:.1f}s) "
            f"— ngủ {sleep:.1f}s"
        )
        time.sleep(sleep)
        waited += sleep

    logger.warning(f"⚠️ {label}Timeout chờ key {provider or 'any'} sau {waited:.0f}s")
    return False


# ── Single provider call với retry thực sự ───────────────────────────

def _annotate_provider(
    batch: List[dict],
    prefer: Optional[str] = None,
    label: str = "",
) -> Optional[Dict[str, List[dict]]]:
    """
    Gọi 1 provider (prefer=None → tự chọn best available).

    Chiến lược retry (FIX v4):
      - prefer=None: thử Groq → Gemini → OpenAI theo thứ tự
      - Mỗi provider: retry MAX_RETRIES lần
      - NoKeyAvailable: CHỜ key hết cooldown rồi thử lại (không skip ngay)
      - Nếu chờ timeout mới chuyển sang provider tiếp theo
    """
    mgr      = get_key_manager()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _user_prompt(batch)},
    ]
    max_tokens = min(len(batch) * TOKENS_PER_REVIEW, MAX_TOKENS_CAP)
    ids        = [item["review_id"] for item in batch]

    providers_to_try = [prefer] if prefer else _PROVIDER_ORDER

    for prov in providers_to_try:
        # Kiểm tra provider này có key nào không (loaded)
        mgr_slots = [s for s in mgr._slots if s.provider == prov]
        if not mgr_slots:
            logger.debug(f"⬛ {label}[{prov}] không có key nào — bỏ qua")
            continue

        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                slot, cfg = mgr.get_slot(provider=prov)
                logger.debug(f"🔑 {label}[{slot}] attempt {attempt}")
                raw    = _call(slot, cfg, messages, max_tokens)
                result = _parse(raw, ids)
                if result:
                    return result
                logger.warning(f"⚠️ {label}[{prov}] parse rỗng attempt {attempt}")
                # Parse rỗng → không cần chờ, retry ngay
                continue

            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429:
                    retry_after = float(
                        e.response.headers.get("retry-after", 5)
                    )
                    mgr.mark_429(slot, retry_after)
                    logger.warning(
                        f"⏳ 429 [{slot}]{label} "
                        f"retry_after={retry_after:.0f}s"
                    )
                    # Chờ key tiếp theo của provider này hết cooldown
                    if not _wait_for_key(prov, label):
                        break   # timeout → sang provider khác
                elif code >= 500:
                    logger.warning(f"❌ HTTP {code} [{slot}]{label} — retry")
                    time.sleep(1.0)
                else:
                    logger.error(
                        f"❌ HTTP {code} [{slot}]{label} — không retry"
                    )
                    break

            except mgr.NoKeyAvailable:
                logger.info(
                    f"⏳ {label}[{prov}] NoKey attempt {attempt} "
                    f"— chờ cooldown..."
                )
                # FIX: chờ đúng thời gian thay vì break ngay
                if not _wait_for_key(prov, label):
                    logger.warning(
                        f"⚠️ {label}[{prov}] timeout chờ — chuyển provider"
                    )
                    break

            except Exception as e:
                logger.warning(f"❌ Unexpected [{prov}]{label}: {e}")
                time.sleep(0.5)

        # Hết MAX_RETRIES cho provider này → sang provider tiếp theo
        logger.warning(
            f"🔁 {label}[{prov}] hết {MAX_RETRIES} retries — thử provider sau"
        )

    logger.error(f"💀 {label} failed — all providers exhausted")
    return None


# Fix: alias exception class để dùng trong except clause
class _NoKeyAvailable(Exception):
    pass


def _annotate_provider_safe(
    batch: List[dict],
    prefer: Optional[str] = None,
    label: str = "",
) -> Optional[Dict[str, List[dict]]]:
    """Wrapper an toàn xử lý exception class đúng cách."""
    mgr      = get_key_manager()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _user_prompt(batch)},
    ]
    max_tokens = min(len(batch) * TOKENS_PER_REVIEW, MAX_TOKENS_CAP)
    ids        = [item["review_id"] for item in batch]

    providers_to_try = [prefer] if prefer else _PROVIDER_ORDER

    for prov in providers_to_try:
        # Bỏ qua provider không có key nào được load
        if not any(s.provider == prov for s in mgr._slots):
            continue

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                slot, cfg = mgr.get_slot(provider=prov)
            except Exception as no_key_exc:
                # NoKeyAvailable — chờ rồi retry
                if "bận" in str(no_key_exc) or "NoKeyAvailable" in type(no_key_exc).__name__:
                    logger.info(
                        f"⏳ {label}[{prov}] NoKey attempt {attempt} — chờ..."
                    )
                    if not _wait_for_key(prov, label):
                        logger.warning(
                            f"⚠️ {label}[{prov}] timeout → next provider"
                        )
                        break   # sang provider tiếp theo
                    continue    # retry cùng provider
                raise

            try:
                logger.debug(f"🔑 {label}[{slot}] attempt {attempt}")
                raw    = _call(slot, cfg, messages, max_tokens)
                result = _parse(raw, ids)
                if result:
                    return result
                logger.warning(
                    f"⚠️ {label}[{prov}] parse rỗng attempt {attempt}"
                )

            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 429:
                    retry_after = float(
                        e.response.headers.get("retry-after", 5)
                    )
                    mgr.mark_429(slot, retry_after)
                    logger.warning(
                        f"⏳ 429 [{slot}]{label} retry_after={retry_after:.0f}s"
                    )
                    if not _wait_for_key(prov, label):
                        break
                elif code >= 500:
                    logger.warning(f"❌ HTTP {code} [{slot}]{label} — retry")
                    time.sleep(1.0)
                else:
                    logger.error(
                        f"❌ HTTP {code} [{slot}]{label} — skip provider"
                    )
                    break

            except Exception as e:
                logger.warning(f"❌ Unexpected [{prov}]{label}: {e}")
                time.sleep(0.5)

        else:
            # Hết MAX_RETRIES mà không success
            pass

        logger.warning(f"🔁 {label}[{prov}] → next provider")

    logger.error(f"💀 {label} all providers failed")
    return None


# ── Ensemble agreement ────────────────────────────────────────────────
def _quad_key(q: dict) -> Tuple[str, str, str]:
    return (q["aspect_term"], q["aspect_category"], q["sentiment"])


def _merge_ensemble(
    results: List[Optional[Dict[str, List[dict]]]],
    ids: List[str],
    n_providers: int,
) -> Dict[str, dict]:
    majority = (n_providers // 2) + 1
    out = {}

    for rid in ids:
        key_to_count: Dict[Tuple, int]  = {}
        key_to_quad:  Dict[Tuple, dict] = {}

        for res in results:
            if res is None:
                continue
            for q in res.get(rid, []):
                k = _quad_key(q)
                key_to_count[k] = key_to_count.get(k, 0) + 1
                key_to_quad[k]  = q

        if not key_to_count:
            out[rid] = {"quadruples": [], "conflict": False}
            continue

        agreed   = [key_to_quad[k] for k, cnt in key_to_count.items() if cnt >= majority]
        minority = [key_to_quad[k] for k, cnt in key_to_count.items() if cnt < majority]

        if agreed:
            out[rid] = {"quadruples": agreed, "conflict": bool(minority)}
        else:
            out[rid] = {"quadruples": list(key_to_quad.values()), "conflict": True}
            logger.debug(f"🔶 [{rid}] Full conflict — union {len(key_to_quad)} quads")

    return out


# ── ThreadPool ────────────────────────────────────────────────────────
_executor: Optional[ThreadPoolExecutor] = None
_exec_lock = threading.Lock()

def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _exec_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_N_WORKERS,
                    thread_name_prefix="annotator",
                )
    return _executor


# ── Public API ────────────────────────────────────────────────────────
def annotate_batch(batch: List[dict]) -> Dict[str, dict]:
    """
    Entry point chính.
    Input : list of {"review_id": str, "text": str}
    Output: {review_id: {"quadruples": [...], "conflict": bool, "provider": str}}
    """
    clean:     List[dict] = []
    noise_ids: List[str]  = []
    for item in batch:
        if _is_noise(item["text"]):
            noise_ids.append(item["review_id"])
        else:
            clean.append(item)

    out: Dict[str, dict] = {
        rid: {"quadruples": [], "conflict": False, "provider": "noise"}
        for rid in noise_ids
    }
    if not clean:
        return out

    ids = [item["review_id"] for item in clean]

    if ENSEMBLE_MODE:
        providers = ["groq", "gemini", "openai"][:ENSEMBLE_N]
        futures: List[Future] = [
            _get_executor().submit(
                _annotate_provider_safe, clean, prov, f"[{prov.upper()}]"
            )
            for prov in providers
        ]
        results  = [f.result() for f in futures]
        merged   = _merge_ensemble(results, ids, n_providers=ENSEMBLE_N)
        for rid in ids:
            entry    = merged.get(rid, {"quadruples": [], "conflict": False})
            out[rid] = {**entry, "provider": "ensemble"}
    else:
        result = _annotate_provider_safe(clean)
        if result is None:
            result = {}
        for rid in ids:
            out[rid] = {
                "quadruples": result.get(rid, []),
                "conflict":   False,
                "provider":   "single",
            }

    return out


def get_current_batch_size() -> int:
    return int(os.getenv("BATCH_SIZE", "10"))