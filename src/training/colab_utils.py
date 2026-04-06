"""
colab_utils.py
==============
Tiện ích chạy trên Google Colab:
  - TeeStream   : ghi ĐỒNG THỜI stdout/stderr ra file và console
  - zip_results : nén tất cả kết quả vào một file .zip
  - download    : tự động tải file về máy qua Colab files API

"""

import os, sys, io, zipfile, shutil
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# ════════════════════════════════════════════════════════════════════════════
# 1. TEE STREAM – ghi stdout & stderr đồng thời ra file
# ════════════════════════════════════════════════════════════════════════════

class _Tee(io.TextIOBase):
    """Ghi đồng thời ra stream gốc và file."""

    def __init__(self, original_stream, file_handle):
        self._orig = original_stream
        self._file = file_handle

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def isatty(self):
        return getattr(self._orig, "isatty", lambda: False)()


class TeeStream:
    """
    Capture toàn bộ stdout và stderr vào một file output_<ts>.txt,
    đồng thời vẫn in ra màn hình Colab như bình thường.

    Sử dụng:
        tee = TeeStream(log_dir="results/logs", ts="20250101_120000")
        ...  # train / evaluate
        tee.restore()   # khôi phục stdout/stderr gốc & đóng file
    """

    def __init__(self, log_dir: str, ts: str | None = None):
        os.makedirs(log_dir, exist_ok=True)
        ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = os.path.join(log_dir, f"output_{ts}.txt")

        self._file      = open(self.out_path, "w", encoding="utf-8", buffering=1)
        self._orig_out  = sys.stdout
        self._orig_err  = sys.stderr

        sys.stdout = _Tee(self._orig_out, self._file)
        sys.stderr = _Tee(self._orig_err, self._file)

        print(f"[TeeStream] Capturing output → {self.out_path}")

    def restore(self):
        """Khôi phục stdout/stderr gốc và đóng file capture."""
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        self._file.flush()
        self._file.close()
        print(f"[TeeStream] Output saved → {self.out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 2. ZIP RESULTS
# ════════════════════════════════════════════════════════════════════════════

def zip_results(
    dirs_to_zip: list[str],
    extra_files: list[str] | None = None,
    output_zip: str | None = None,
) -> str:
    """
    Nén các thư mục/file kết quả vào một file .zip duy nhất.

    Args:
        dirs_to_zip  : Danh sách thư mục cần nén (vd: ["models/bilstm", "results"]).
        extra_files  : Danh sách file lẻ cần thêm vào zip (vd: log, json).
        output_zip   : Đường dẫn file zip đầu ra. Mặc định "bilstm_<ts>.zip".

    Returns:
        Đường dẫn file zip đã tạo.
    """
    if output_zip is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_zip = f"bilstm_{ts}.zip"

    extra_files = extra_files or []
    paths_added  = 0
    skipped      = 0

    print(f"\n📦 Đang nén kết quả → {output_zip} ...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── Nén từng thư mục ─────────────────────────────────────────────
        for dir_path in dirs_to_zip:
            if not os.path.isdir(dir_path):
                print(f"   ⚠️  Bỏ qua (không tìm thấy): {dir_path}")
                skipped += 1
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    fpath    = os.path.join(root, fname)
                    arcname  = fpath          # giữ nguyên đường dẫn tương đối
                    zf.write(fpath, arcname)
                    paths_added += 1

        # ── Thêm file lẻ ─────────────────────────────────────────────────
        for fpath in extra_files:
            if not os.path.isfile(fpath):
                print(f"   ⚠️  Bỏ qua (không tìm thấy): {fpath}")
                skipped += 1
                continue
            zf.write(fpath, fpath)
            paths_added += 1

    size_mb = os.path.getsize(output_zip) / 1_048_576
    print(f"   ✅ {paths_added} file, bỏ qua {skipped} → "
          f"{output_zip} ({size_mb:.1f} MB)")
    return output_zip


# ════════════════════════════════════════════════════════════════════════════
# 3. AUTO DOWNLOAD (Colab)
# ════════════════════════════════════════════════════════════════════════════

def download_file(path: str):
    """
    Tải file về máy qua google.colab.files.
    Nếu không chạy trong Colab thì in đường dẫn file và bỏ qua.
    """
    try:
        from google.colab import files  # type: ignore
        print(f"\n⬇️  Đang tải {path} về máy ...")
        files.download(path)
        print("   ✅ Tải xong!")
    except ImportError:
        print(f"\n[Không phải Colab] File nằm tại: {os.path.abspath(path)}")
    except Exception as e:
        print(f"\n⚠️  Lỗi khi tải: {e}")
        print(f"   File nằm tại: {os.path.abspath(path)}")


# ════════════════════════════════════════════════════════════════════════════
# 4. HÀM GỘP: zip rồi download luôn
# ════════════════════════════════════════════════════════════════════════════

def zip_and_download(
    dirs_to_zip: list[str],
    extra_files: list[str] | None = None,
    output_name: str | None = None,
):
    """
    Nén kết quả rồi tự động tải về máy.
    Gọi ở cuối hàm train() sau khi tee.restore().

    Args:
        dirs_to_zip  : Các thư mục cần nén.
        extra_files  : File lẻ bổ sung (log, txt, …).
        output_name  : Tên file zip (mặc định tự sinh theo timestamp).
    """
    zip_path = zip_results(dirs_to_zip, extra_files, output_name)
    download_file(zip_path)