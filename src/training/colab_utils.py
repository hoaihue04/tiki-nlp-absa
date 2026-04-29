#!/usr/bin/env python3
"""
colab_utils.py — Tiện ích cho Google Colab
===========================================
Giải quyết vấn đề Colab mất kết nối / runtime reset:
    1. Auto-mount Google Drive
    2. Sync checkpoints lên Drive sau mỗi epoch
    3. Khôi phục từ Drive khi restart
    4. Keep-alive script chống timeout

Cách sử dụng trong Colab notebook:
    from src.training.colab_utils import setup_colab, ColabTrainingWrapper

    # Bước 1: Setup (chạy 1 lần đầu)
    setup_colab(project_name='tiki-absa', drive_folder='MyDrive/tiki_absa')

    # Bước 2: Chạy training (tự resume nếu có checkpoint)
    !python src/training/train_phobert.py

Hoặc dùng trực tiếp trong cell:
    import sys
    sys.path.insert(0, '/content/tiki')
    from src.training.colab_utils import ColabAutoSync
    sync = ColabAutoSync(local_dir='checkpoints/phobert',
                         drive_dir='/content/drive/MyDrive/tiki_absa/phobert')
"""

import os
import sys
import shutil
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════
# 1. GOOGLE DRIVE SETUP
# ══════════════════════════════════════════════════════════════════════════
def mount_drive(mount_point: str = '/content/drive') -> bool:
    """
    Mount Google Drive trong Colab.
    Trả về True nếu thành công, False nếu không phải Colab.
    """
    try:
        from google.colab import drive
        if not os.path.exists(os.path.join(mount_point, 'MyDrive')):
            print('[Colab] Mounting Google Drive...')
            drive.mount(mount_point)
        else:
            print('[Colab] Google Drive đã được mount.')
        return True
    except ImportError:
        print('[INFO] Không phải môi trường Colab — bỏ qua mount Drive.')
        return False


def is_colab() -> bool:
    """Kiểm tra có đang chạy trong Google Colab không."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════
# 2. AUTO SYNC: LOCAL ↔ DRIVE
# ══════════════════════════════════════════════════════════════════════════
class ColabAutoSync:
    """
    Sync thư mục checkpoint giữa local Colab storage và Google Drive.
    Sau mỗi epoch, gọi .sync() để đồng bộ.
    Khi restart, gọi .restore() để khôi phục từ Drive.
    """
    def __init__(self, local_dir: str, drive_dir: str):
        self.local_dir = local_dir
        self.drive_dir = drive_dir

    def sync(self):
        """Sync local → Drive (upload checkpoint mới nhất)."""
        if not os.path.exists(self.drive_dir):
            os.makedirs(self.drive_dir, exist_ok=True)
        if not os.path.exists(self.local_dir):
            return
        try:
            # Sync toàn bộ thư mục
            self._sync_dir(self.local_dir, self.drive_dir)
            print(f'  [Drive] Sync ✓ → {self.drive_dir}')
        except Exception as e:
            print(f'  [Drive] Sync lỗi: {e}')

    def restore(self):
        """Khôi phục từ Drive → local (sau khi restart)."""
        if not os.path.exists(self.drive_dir):
            print(f'  [Drive] Không tìm thấy {self.drive_dir} trên Drive.')
            return False
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            self._sync_dir(self.drive_dir, self.local_dir)
            print(f'  [Drive] Restore ✓ ← {self.drive_dir}')
            return True
        except Exception as e:
            print(f'  [Drive] Restore lỗi: {e}')
            return False

    def _sync_dir(self, src: str, dst: str):
        """Copy tất cả file từ src → dst (chỉ copy nếu mới hơn)."""
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(dst, item)
            if os.path.isdir(src_path):
                self._sync_dir(src_path, dst_path)
            else:
                # Copy nếu dst không tồn tại hoặc src mới hơn
                if (not os.path.exists(dst_path) or
                        os.path.getmtime(src_path) > os.path.getmtime(dst_path)):
                    shutil.copy2(src_path, dst_path)


# ══════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT MANAGER (dùng trong training scripts)
# ══════════════════════════════════════════════════════════════════════════
class CheckpointManager:
    """
    Quản lý checkpoint cho Colab training.
    - Tự động tìm checkpoint mới nhất
    - Giữ N checkpoint gần nhất (tiết kiệm dung lượng)
    - Sync lên Drive sau mỗi lần lưu
    """
    def __init__(self, checkpoint_dir: str, keep_last: int = 3,
                 drive_sync: Optional[ColabAutoSync] = None):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last      = keep_last
        self.drive_sync     = drive_sync
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, state: dict, epoch: int):
        """Lưu checkpoint và xóa các checkpoint cũ."""
        import torch
        path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pt')
        torch.save(state, path)
        print(f'  [Checkpoint] Saved epoch {epoch} → {path}')

        # Xóa checkpoint cũ, chỉ giữ keep_last
        self._cleanup()

        # Sync lên Drive
        if self.drive_sync is not None:
            self.drive_sync.sync()

    def load_latest(self):
        """Load checkpoint mới nhất, trả về (state, epoch) hoặc (None, 0)."""
        import torch
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith('epoch_') and f.endswith('.pt')]
        if not files:
            return None, 0
        epochs = [int(f.replace('epoch_', '').replace('.pt', '')) for f in files]
        latest = max(epochs)
        path   = os.path.join(self.checkpoint_dir, f'epoch_{latest}.pt')
        print(f'  [Checkpoint] Resume từ epoch {latest}: {path}')
        state  = torch.load(path, map_location='cpu')
        return state, latest

    def _cleanup(self):
        """Giữ lại keep_last checkpoint, xóa phần còn lại."""
        files = [f for f in os.listdir(self.checkpoint_dir)
                 if f.startswith('epoch_') and f.endswith('.pt')]
        if len(files) <= self.keep_last:
            return
        epochs = sorted([int(f.replace('epoch_', '').replace('.pt', ''))
                         for f in files])
        to_delete = epochs[:-self.keep_last]
        for ep in to_delete:
            path = os.path.join(self.checkpoint_dir, f'epoch_{ep}.pt')
            if os.path.exists(path):
                os.remove(path)


# ══════════════════════════════════════════════════════════════════════════
# 4. SETUP FUNCTION (gọi 1 lần ở đầu notebook)
# ══════════════════════════════════════════════════════════════════════════
def setup_colab(project_name: str = 'tiki-absa',
                drive_folder: str = 'MyDrive/tiki_absa',
                repo_url: Optional[str] = None,
                install_deps: bool = True):
    """
    Setup Colab environment đầy đủ.
    Gọi ở cell đầu tiên của notebook.

    Args:
        project_name: Tên thư mục project trên Colab (/content/<project_name>)
        drive_folder: Đường dẫn folder trên Drive (sau /content/drive/)
        repo_url    : Git repo URL để clone (None = bỏ qua)
        install_deps: Cài đặt pip dependencies không?
    """
    print('=' * 55)
    print('  COLAB SETUP — Tiki ABSA')
    print('=' * 55)

    # 1. Mount Drive
    mounted = mount_drive()

    # 2. Clone repo (nếu có)
    if repo_url:
        project_path = f'/content/{project_name}'
        if not os.path.exists(project_path):
            print(f'\n[2] Clone repo: {repo_url}')
            subprocess.run(['git', 'clone', repo_url, project_path], check=True)
        else:
            print(f'\n[2] Repo đã tồn tại: {project_path}')
        os.chdir(project_path)
        sys.path.insert(0, project_path)
        print(f'    Working dir: {os.getcwd()}')

    # 3. Install dependencies
    if install_deps:
        print('\n[3] Cài đặt dependencies...')
        deps = [
            'torch', 'transformers', 'sentencepiece',
            'torchcrf', 'scikit-learn', 'numpy', 'pandas',
        ]
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + deps,
            check=False
        )
        print('    Done.')

    # 4. Restore checkpoints từ Drive
    if mounted:
        drive_base = f'/content/drive/{drive_folder}'
        print(f'\n[4] Restore checkpoints từ Drive: {drive_base}')
        for model_name in ['bilstm', 'phobert', 'vit5']:
            local_ckpt = f'checkpoints/{model_name}'
            drive_ckpt = f'{drive_base}/checkpoints/{model_name}'
            if os.path.exists(drive_ckpt):
                syncer = ColabAutoSync(local_ckpt, drive_ckpt)
                syncer.restore()
            # Restore models
            local_model  = f'models/{model_name}'
            drive_model  = f'{drive_base}/models/{model_name}'
            if os.path.exists(drive_model):
                syncer = ColabAutoSync(local_model, drive_model)
                syncer.restore()

    print('\n  Setup hoàn tất!')
    print('\nBước tiếp theo:')
    print('  !python src/training/prepare_data.py')
    print('  !python src/training/train_phobert.py')
    print('  !python src/training/train_vit5.py')
    print('  !python src/training/train_bilstm.py')


# ══════════════════════════════════════════════════════════════════════════
# 5. KEEP-ALIVE (chống Colab timeout)
# ══════════════════════════════════════════════════════════════════════════
def start_keepalive():
    """
    Khởi động keep-alive loop trong background thread.
    Gọi ở đầu notebook để tránh Colab timeout.

    Chú ý: Colab Pro/Pro+ ít bị timeout hơn.
    Với free Colab, script này KHÔNG đảm bảo 100% không bị ngắt,
    nhưng checkpoint sẽ giúp tiếp tục từ điểm dừng.
    """
    import threading

    def _keepalive():
        while True:
            time.sleep(60 * 5)  # 5 phút một lần
            try:
                # Trigger nhẹ để giữ session
                _ = os.getcwd()
            except Exception:
                break

    t = threading.Thread(target=_keepalive, daemon=True)
    t.start()
    print('[KeepAlive] Started (ping mỗi 5 phút)')


# ══════════════════════════════════════════════════════════════════════════
# 6. HƯỚNG DẪN SỬ DỤNG COLAB
# ══════════════════════════════════════════════════════════════════════════
COLAB_GUIDE = """
╔══════════════════════════════════════════════════════════╗
║          HƯỚNG DẪN TRAIN TRÊN GOOGLE COLAB              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Cell 1 — Setup (chạy 1 lần)                            ║
║  ─────────────────────────                              ║
║  from google.colab import drive                         ║
║  drive.mount('/content/drive')                          ║
║  !git clone <repo_url> /content/tiki                    ║
║  %cd /content/tiki                                      ║
║  !pip install -q transformers sentencepiece torchcrf    ║
║                                                         ║
║  Cell 2 — Chuẩn bị dữ liệu (chạy 1 lần)               ║
║  ─────────────────────────────────────                  ║
║  !python src/training/prepare_data.py                   ║
║                                                         ║
║  Cell 3 — Train PhoBERT                                 ║
║  ─────────────────────                                  ║
║  !python src/training/train_phobert.py                  ║
║  # Nếu mất kết nối → Runtime > Run all → tự resume     ║
║                                                         ║
║  Cell 4 — Train ViT5                                    ║
║  ────────────────────                                   ║
║  !python src/training/train_vit5.py                     ║
║                                                         ║
║  Cell 5 — Train BiLSTM                                  ║
║  ─────────────────────                                  ║
║  !python src/training/train_bilstm.py                   ║
║                                                         ║
║  Cell 6 — Sync kết quả lên Drive                       ║
║  ──────────────────────────────                         ║
║  import shutil                                          ║
║  shutil.copytree('checkpoints',                         ║
║      '/content/drive/MyDrive/tiki_absa/checkpoints',   ║
║      dirs_exist_ok=True)                                ║
║  shutil.copytree('models',                              ║
║      '/content/drive/MyDrive/tiki_absa/models',         ║
║      dirs_exist_ok=True)                                ║
║                                                         ║
║  LƯU Ý:                                                 ║
║  - Checkpoint lưu mỗi epoch → mất kết nối không mất    ║
║    công train                                           ║
║  - Chạy lại cell tương ứng là tiếp tục tự động          ║
║  - Colab Pro khuyến nghị để tránh timeout               ║
║  - T4 GPU free: PhoBERT ~30ph/epoch, ViT5 ~45ph/epoch  ║
║  - A100 Pro: PhoBERT ~8ph/epoch, ViT5 ~12ph/epoch      ║
╚══════════════════════════════════════════════════════════╝
"""


def print_guide():
    """In hướng dẫn sử dụng Colab."""
    print(COLAB_GUIDE)


# ══════════════════════════════════════════════════════════════════════════
# MAIN — In hướng dẫn khi chạy trực tiếp
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print_guide()
    print('\nChạy trong Colab:')
    print('  from src.training.colab_utils import setup_colab')
    print('  setup_colab(project_name="tiki", drive_folder="MyDrive/tiki_absa")')
