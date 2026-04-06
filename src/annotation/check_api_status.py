import time
import requests
from config import get_key_manager, PROVIDERS

# ─────────────────────────────────────────────
# ⚙️ CONFIG
# ─────────────────────────────────────────────
TIMEOUT = 6
AUTO_COOLDOWN = True
COOLDOWN_SECONDS = 60


# ─────────────────────────────────────────────
# 🧪 TEST 1 KEY
# ─────────────────────────────────────────────
def test_key(slot, cfg):
    try:
        response = requests.post(
            cfg["url"],
            headers={
                "Authorization": f"Bearer {slot.key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg["model"],
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1
            },
            timeout=TIMEOUT
        )

        status = response.status_code

        # ✅ OK
        if status == 200:
            return "OK"

        # 🔴 Unauthorized / key chết
        elif status in [401, 403]:
            return "INVALID_KEY"

        # ⚠️ Rate limit
        elif status == 429:
            return "RATE_LIMIT"

        # ❓ lỗi khác
        else:
            return f"ERROR_{status}"

    except requests.exceptions.Timeout:
        return "TIMEOUT"

    except Exception as e:
        return "EXCEPTION"


# ─────────────────────────────────────────────
# 🔍 CHECK ALL KEYS
# ─────────────────────────────────────────────
def check_all_keys():
    mgr = get_key_manager()

    print("\n🔍 CHECK API KEYS HEALTH")
    print("=" * 60)

    results = {
        "OK": 0,
        "INVALID_KEY": 0,
        "RATE_LIMIT": 0,
        "TIMEOUT": 0,
        "EXCEPTION": 0,
        "OTHER": 0,
    }

    for slot in mgr._slots:
        cfg = PROVIDERS[slot.provider]

        status = test_key(slot, cfg)

        # 📊 đếm
        if status in results:
            results[status] += 1
        else:
            results["OTHER"] += 1

        # 🧠 xử lý cooldown nếu cần
        if AUTO_COOLDOWN and status in ["INVALID_KEY", "RATE_LIMIT", "TIMEOUT"]:
            slot.cool(COOLDOWN_SECONDS)

        # 🖨️ output đẹp
        emoji = {
            "OK": "✅",
            "INVALID_KEY": "❌",
            "RATE_LIMIT": "⏳",
            "TIMEOUT": "⌛",
            "EXCEPTION": "💥"
        }.get(status, "❓")

        print(f"{emoji} [{slot.provider}#{slot.index}] → {status}")

        time.sleep(0.2)  # tránh spam API

    # ─────────────────────────────
    # 📊 SUMMARY
    # ─────────────────────────────
    print("\n📊 SUMMARY")
    print("=" * 60)

    for k, v in results.items():
        print(f"{k:15}: {v}")

    print("=" * 60)
    print(f"🚀 Total theoretical RPM: ~{mgr.total_theoretical_rpm()}")
    print(f"🔑 Available now: {mgr.available_count()}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    check_all_keys()