
# src/annotation/run_annotation.py
"""
Cách dùng:
  python run_annotation.py                    # default batch
  python run_annotation.py --batch 500        # đúng 500 reviews
  python run_annotation.py --batch 0          # toàn bộ còn lại
  python run_annotation.py --reset            # xóa checkpoint & output
  python run_annotation.py --dry-run          # xem plan, không gọi API
  python run_annotation.py --stats            # chỉ in thống kê checkpoint
  ENSEMBLE_MODE=true python run_annotation.py # dùng 2 providers + voting
────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Set


import time
import logging
from config import get_key_manager, PROVIDERS



sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    INPUT_FILE, OUTPUT_JSONL, OUTPUT_CSV, CHECKPOINT_FILE,
    DEFAULT_BATCH_SIZE, CHECKPOINT_INTERVAL, MAX_REVIEW_LENGTH,
)
from annotator import annotate_batch, get_current_batch_size
from config import get_key_manager, PROVIDERS


# ── Paths ────────────────────────────────────────────────────
_OUT_DIR        = Path(OUTPUT_JSONL).parent
SKIPPED_JSONL   = _OUT_DIR / "skipped_reviews.jsonl"
CONFLICT_JSONL  = _OUT_DIR / "conflict_reviews.jsonl"   # cần human review
LOG_FILE        = _OUT_DIR / "annotation.log"
STATUS_EVERY    = 50     # in provider status mỗi N reviews


# ── Logging ──────────────────────────────────────────────────
_OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class SmartThrottle:
    """
    Điều tiết tốc độ gửi batch dựa trên trạng thái key thực tế.

    Logic:
      1. Nếu có key available → tính min_gap theo RPM, chờ nếu gửi quá nhanh
      2. Nếu KHÔNG có key nào → chờ đến khi key gần nhất hết cooldown
         (thay vì chờ fixed 2s rồi fail)
    """

    MAX_WAIT    = 70.0    # giây — cap tuyệt đối, tránh treo vĩnh viễn
    BUFFER_PCTS = 1.20    # +20% buffer trên min_gap

    def __init__(self):
        self._last          = 0.0
        self._no_key_streak = 0

    # ── Internal ──────────────────────────────────────────────────────

    def _soonest_key_free(self) -> float:
        """
        Số giây cho đến khi có key nào đó available.
        0.0 nếu đã có key sẵn.
        """
        mgr  = get_key_manager()
        now  = time.monotonic()
        best = None   # None = chưa thấy key nào

        for s in mgr._slots:
            if s.is_available:
                return 0.0  # đã có key free ngay bây giờ

            # Cooldown chưa hết
            cd = s._cooldown_until
            if cd > now:
                wait = cd - now
            else:
                # Key không trong cooldown nhưng đang đầy RPM window
                # → ước tính: request cũ nhất trong window sẽ expire sau bao lâu
                with s._lock:
                    s._evict()
                    if s._req_times:
                        oldest = s._req_times[0]
                        wait   = max(0.0, (oldest + 60.0) - now)
                    else:
                        wait = 0.0

            if best is None or wait < best:
                best = wait

        return min(best if best is not None else self.MAX_WAIT, self.MAX_WAIT)

    # ── Public ────────────────────────────────────────────────────────

    def wait(self):
        """Gọi trước mỗi batch. Block cho đến khi an toàn để gửi."""
        mgr   = get_key_manager()
        avail = mgr.available_count()

        if avail == 0:
            # ── Không có key nào: chờ đến khi key gần nhất free ──────
            self._no_key_streak += 1
            wait_needed = self._soonest_key_free()

            if wait_needed <= 0.0:
                # Có key free ngay (race condition), tiếp tục
                self._no_key_streak = 0
                return

            # Chờ chính xác, nhắc log mỗi 5s
            logger.info(
                f"⏳ Tất cả keys bận — chờ {wait_needed:.1f}s "
                f"(streak={self._no_key_streak})"
            )
            waited = 0.0
            while waited < wait_needed and waited < self.MAX_WAIT:
                chunk = min(5.0, wait_needed - waited)
                time.sleep(chunk)
                waited += chunk
                if mgr.available_count() > 0:
                    break   # có key free sớm hơn dự kiến
            self._no_key_streak = 0
            return

        # ── Có key: throttle theo RPM ──────────────────────────────
        self._no_key_streak = 0

        # Tính RPM của các key available
        slots_available = [
            s for s in mgr._slots if s.is_available
        ]
        # Tổng RPM = sum(rpm_limit) của keys available
        total_rpm = sum(s.rpm_limit for s in slots_available)
        if total_rpm == 0:
            return

        # min_gap = 60s / total_rpm, với buffer
        min_gap = (60.0 / total_rpm) * self.BUFFER_PCTS

        elapsed   = time.monotonic() - self._last
        remaining = min_gap - elapsed
        if remaining > 0.05:
            time.sleep(remaining)

        self._last = time.monotonic()

# ── Checkpoint ───────────────────────────────────────────────
_ckpt = Path(CHECKPOINT_FILE)

def load_checkpoint() -> Set[str]:
    if not _ckpt.exists():
        return set()
    return {l.strip() for l in _ckpt.read_text("utf-8").splitlines() if l.strip()}

def save_checkpoint_atomic(ids: Set[str]):
    tmp = _ckpt.with_suffix(".tmp")
    tmp.write_text("\n".join(sorted(ids)), encoding="utf-8")
    tmp.replace(_ckpt)

def append_checkpoint(rid: str):
    with open(_ckpt, "a", encoding="utf-8") as f:
        f.write(rid + "\n")


# ── Buffered JSONL writer ────────────────────────────────────
class JWriter:
    def __init__(self, path):
        self.path = Path(path)
        self._buf: List[str] = []

    def write(self, rec: dict):
        self._buf.append(json.dumps(rec, ensure_ascii=False))

    def flush(self):
        if not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.flush()


# ── CSV writer ───────────────────────────────────────────────
def write_csv(csv_path: str, rid: str, text: str, quads: list):
    exists = Path(csv_path).exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "review_id", "text_preview",
            "aspect_term", "aspect_category", "opinion_term", "sentiment",
        ])
        if not exists:
            w.writeheader()
        preview = text[:100].replace("\n", " ")
        for q in quads:
            w.writerow({**q, "review_id": rid, "text_preview": preview})


# ── Data loader ──────────────────────────────────────────────
def load_reviews() -> List[dict]:
    reviews = []
    with open(INPUT_FILE, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid  = str(row.get("review_id", "")).strip()
            text = (
                row.get("normalized_text")
                or row.get("cleaned_content")
                or row.get("content", "")
            ).strip()
            if rid and text:
                reviews.append({"review_id": rid, "text": text[:MAX_REVIEW_LENGTH * 4]})
    return reviews


# ── Signal handler ───────────────────────────────────────────
_done_ids: Set[str] = set()
_jwriter: JWriter   = None   # type: ignore

def _on_sigint(sig, frame):
    logger.warning("\n🛑 Interrupt → flushing + saving checkpoint...")
    if _jwriter:
        _jwriter.flush()
    save_checkpoint_atomic(_done_ids)
    logger.info("✅ Checkpoint saved. Thoát an toàn.")
    sys.exit(0)


# ── Args ─────────────────────────────────────────────────────
def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH_SIZE,
                   help="Số reviews cần xử lý (0 = tất cả)")
    p.add_argument("--reset",   action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stats",   action="store_true")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────
def main():
    global _done_ids, _jwriter

    args = _args()
    signal.signal(signal.SIGINT, _on_sigint)

    # Reset
    if args.reset:
        for path in [OUTPUT_JSONL, OUTPUT_CSV, CHECKPOINT_FILE,
                     str(SKIPPED_JSONL), str(CONFLICT_JSONL)]:
            if Path(path).exists():
                Path(path).unlink()
                logger.info(f"🗑  Đã xóa: {path}")

    all_reviews = load_reviews()
    _done_ids   = load_checkpoint()
    pending     = [r for r in all_reviews if r["review_id"] not in _done_ids]

    if args.batch and args.batch > 0:
        pending = pending[: args.batch]

    logger.info(
        f"📊 Tổng: {len(all_reviews):,} | "
        f"Đã xử lý: {len(_done_ids):,} | "
        f"Còn lại: {len(pending):,}"
    )

    # Khởi tạo KeyManager → in key status
    mgr = get_key_manager()
    logger.info(f"🔑 Tổng lý thuyết: ~{mgr.total_theoretical_rpm()} RPM")
    for line in mgr.status_lines():
        logger.info(line)

    ensemble = os.getenv("ENSEMBLE_MODE", "false").lower() == "true"
    if ensemble:
        logger.info("🔀 ENSEMBLE MODE bật — Groq + Gemini song song")

    if args.stats or not pending:
        if not pending:
            logger.info("🎉 Không còn review nào cần xử lý!")
        return

    if args.dry_run:
        bsz = get_current_batch_size()
        n   = (len(pending) + bsz - 1) // bsz
        logger.info(f"🔍 DRY RUN: {len(pending)} reviews | batch={bsz} | {n} batches")
        return

    # ── Main loop ────────────────────────────────────────────
    throttle = SmartThrottle()
    success  = 0
    skipped  = 0
    conflicts= 0
    t0       = time.monotonic()
    processed= 0

    with (
        JWriter(OUTPUT_JSONL)       as jw,
        JWriter(str(SKIPPED_JSONL)) as sw,
        JWriter(str(CONFLICT_JSONL))as cw,
    ):
        _jwriter = jw   # để signal handler có thể flush

        i = 0
        while i < len(pending):
            bsz   = get_current_batch_size()
            chunk = pending[i: i + bsz]

            throttle.wait()   # ← điều tiết thông minh, không 429

            results = annotate_batch(chunk)

            for item in chunk:
                rid  = item["review_id"]
                text = item["text"]
                res  = results.get(rid, {"quadruples": [], "conflict": False})
                quads    = res["quadruples"]
                conflict = res.get("conflict", False)

                # Ghi JSONL chính
                jw.write({
                    "review_id":  rid,
                    "text":       text,
                    "quadruples": quads,
                    "conflict":   conflict,
                })

                if quads:
                    success += 1
                    write_csv(OUTPUT_CSV, rid, text, quads)
                    if conflict:
                        conflicts += 1
                        cw.write({"review_id": rid, "text": text,
                                  "quadruples": quads, "reason": "MODEL_DISAGREEMENT"})
                else:
                    skipped += 1
                    sw.write({"review_id": rid, "text": text, "reason": "NO_ASPECT"})

                _done_ids.add(rid)
                append_checkpoint(rid)

            processed += len(chunk)
            i += bsz

            # Progress
            elapsed = time.monotonic() - t0
            rate    = processed / elapsed if elapsed else 0
            pct     = processed / len(pending) * 100
            eta     = (len(pending) - processed) / rate if rate else 0
            eta_s   = f"{eta/60:.1f}m" if eta > 90 else f"{eta:.0f}s"

            logger.info(
                f"[{processed:>{len(str(len(pending)))}}/{len(pending)}] "
                f"{pct:5.1f}% | ✅{success} ⬜{skipped} 🔶{conflicts} "
                f"| {rate:.2f} rev/s | ETA {eta_s}"
            )

            # Checkpoint định kỳ
            if processed % CHECKPOINT_INTERVAL == 0:
                jw.flush()
                save_checkpoint_atomic(_done_ids)

            # Provider status định kỳ
            if processed % STATUS_EVERY == 0:
                avail = mgr.available_count()
                total = len(mgr._slots)
                logger.info(f"🔑 Keys available: {avail}/{total}")

    save_checkpoint_atomic(_done_ids)

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    total   = success + skipped
    logger.info("=" * 60)
    logger.info("✅ HOÀN THÀNH")
    logger.info(f"⏱  Thời gian     : {elapsed/60:.1f} phút")
    logger.info(f"✔  Có quadruples : {success:,}")
    logger.info(f"✖  Không có quad : {skipped:,}  ({skipped/total*100:.1f}%)" if total else "")
    logger.info(f"🔶 Conflict (ensemble): {conflicts:,}")
    logger.info(f"📁 JSONL  : {OUTPUT_JSONL}")
    logger.info(f"📁 CSV    : {OUTPUT_CSV}")
    logger.info(f"📁 Skipped: {SKIPPED_JSONL}")
    if conflicts:
        logger.info(f"📁 Conflict: {CONFLICT_JSONL}  ← cần verify thủ công")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()