"""
config.py — Single-Provider Key Manager (Groq only)

"""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ── Provider configs ──────────────────────────────────────────────────
# NOTE: Chỉ giữ Groq. Gemini & OpenAI bị tắt để tránh fallback không mong muốn.
# Để bật lại: thêm "gemini" / "openai" vào dict này và thêm key vào .env
PROVIDERS = {
    "groq": {
        "url":      "https://api.groq.com/openai/v1/chat/completions",
        "model":    "llama-3.3-70b-versatile",
        "rpm":      28,        
        "tpm":      14_000,
        "priority": 1,
    },
}


# ── Per-key slot ──────────────────────────────────────────────────────
@dataclass
class KeySlot:
    provider: str
    key:      str
    index:    int

    _lock:           threading.Lock = field(default_factory=threading.Lock, repr=False)
    _req_times:      deque          = field(default_factory=deque,          repr=False)
    _cooldown_until: float          = field(default=0.0,                    repr=False)

    rpm_limit: int = field(default=28, repr=False)

    def _evict(self):
        """Xóa timestamps ngoài cửa sổ 60s."""
        cutoff = time.monotonic() - 60.0
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()

    @property
    def requests_in_window(self) -> int:
        with self._lock:
            self._evict()
            return len(self._req_times)

    @property
    def is_available(self) -> bool:
        with self._lock:
            if time.monotonic() < self._cooldown_until:
                return False
            self._evict()
            return len(self._req_times) < self.rpm_limit

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    def record(self):
        with self._lock:
            self._req_times.append(time.monotonic())

    def cool(self, seconds: float):
        with self._lock:
            self._cooldown_until = time.monotonic() + seconds
        logger.warning(f"🔴 [{self.provider}#{self.index}] cooldown {seconds:.0f}s")

    def __repr__(self):
        return f"<{self.provider}#{self.index} {self.requests_in_window}/{self.rpm_limit}>"


# ── Multi-provider manager ────────────────────────────────────────────
class KeyManager:
    """
    get_slot()               → best available (Groq > Gemini > OpenAI)
    get_slot("groq")         → force Groq only
    get_slot("gemini")       → force Gemini only
    get_slot("openai")       → force OpenAI only
    """

    class NoKeyAvailable(RuntimeError):
        pass

    def __init__(self):
        self._slots: List[KeySlot] = []
        self._lock  = threading.Lock()
        self._load()

    # ── Internal ──────────────────────────────────────────────────────
    def _load(self):
        for provider, cfg in PROVIDERS.items():
            keys = self._env_keys(provider)
            for i, k in enumerate(keys, 1):
                self._slots.append(KeySlot(
                    provider=provider, key=k, index=i,
                    rpm_limit=cfg["rpm"],
                ))
            if keys:
                logger.info(
                    f"✅ {provider.upper():8s}: {len(keys)} key(s) "
                    f"→ ~{len(keys) * cfg['rpm']} RPM  ({cfg['model']})"
                )
            else:
                logger.debug(f"⬛ {provider.upper()}: no keys found (optional)")

        if not self._slots:
            raise RuntimeError(
                "❌ Không tìm thấy API key nào.\n"
                "→ Kiểm tra file .env:\n"
                "    GROQ_API_KEY_1=...   GROQ_API_KEY_2=...\n"
                "    GEMINI_API_KEY_1=... GEMINI_API_KEY_2=...\n"
                "    OPENAI_API_KEY_1=... OPENAI_API_KEY_2=...\n"
                "  (cũng nhận GROQ_API_KEY không có số)"
            )

        logger.info("🔑 Loaded keys:")
        for s in self._slots:
            logger.info(f"   {s.provider}#{s.index}")

    @staticmethod
    def _env_keys(provider: str) -> List[str]:
        """
        Đọc cả 2 kiểu:
          GROQ_API_KEY          (không số)
          GROQ_API_KEY_1 ... _N (có số)
        """
        base = f"{provider.upper()}_API_KEY"
        keys: List[str] = []
        seen: set        = set()

        # Dạng không số: GROQ_API_KEY
        k0 = os.getenv(base, "").strip()
        if k0 and k0 not in seen:
            keys.append(k0)
            seen.add(k0)

        # Dạng có số: GROQ_API_KEY_1 ... GROQ_API_KEY_19
        for i in range(1, 20):
            k = os.getenv(f"{base}_{i}", "").strip()
            if k and k not in seen:
                keys.append(k)
                seen.add(k)

        return keys

    # ── Public API ────────────────────────────────────────────────────
    def get_slot(self, provider: Optional[str] = None) -> Tuple["KeySlot", dict]:
        """
        Trả về (slot, provider_cfg) tốt nhất.

        Ưu tiên: Groq (priority=1) → Gemini (priority=2) → OpenAI (priority=3)
        Trong cùng provider: chọn key ít request nhất.
        """
        with self._lock:
            candidates = [
                s for s in self._slots
                if s.is_available and (provider is None or s.provider == provider)
            ]
            if not candidates:
                raise KeyManager.NoKeyAvailable(
                    f"Tất cả keys {'(' + provider + ')' if provider else ''} đang bận"
                )
            candidates.sort(key=lambda s: (
                PROVIDERS[s.provider]["priority"],
                s.requests_in_window,
            ))
            best = candidates[0]
            best.record()
            return best, PROVIDERS[best.provider]

    def mark_429(self, slot: "KeySlot", retry_after: float = 5.0):
        slot.cool(retry_after)

    def available_count(self, provider: Optional[str] = None) -> int:
        return sum(
            1 for s in self._slots
            if s.is_available and (provider is None or s.provider == provider)
        )

    def status_lines(self) -> List[str]:
        lines = []
        for s in self._slots:
            avail = "✅" if s.is_available else f"⏳{s.cooldown_remaining:.0f}s"
            lines.append(
                f"  [{s.provider}#{s.index}] "
                f"{s.requests_in_window}/{s.rpm_limit} RPM {avail}"
            )
        return lines

    def total_theoretical_rpm(self) -> int:
        total = 0
        for provider, cfg in PROVIDERS.items():
            n = sum(1 for s in self._slots if s.provider == provider)
            total += n * cfg["rpm"]
        return total


# ── Singleton ─────────────────────────────────────────────────────────
_manager:  Optional[KeyManager] = None
_mgr_lock: threading.Lock       = threading.Lock()

def get_key_manager() -> KeyManager:
    global _manager
    if _manager is None:
        with _mgr_lock:
            if _manager is None:
                _manager = KeyManager()
    return _manager