"""
run_all.py
==========
Script tiện lợi để chạy toàn bộ pipeline từ đầu đến cuối.
Dành cho người mới bắt đầu.

Cách chạy:
  cd TIKI
  python run_all.py --step prepare          # chỉ chuẩn bị dữ liệu
  python run_all.py --step train_bilstm     # train BiLSTM
  python run_all.py --step train_phobert    # train PhoBERT
  python run_all.py --step train_vit5       # train ViT5
  python run_all.py --step compare          # so sánh models
  python run_all.py --step demo             # chạy web demo
  python run_all.py --all                   # chạy tất cả (trừ demo)
"""

import subprocess, sys, argparse, os

PYTHON = sys.executable


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"▶️  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Thất bại! Kiểm tra lỗi ở trên.")
        sys.exit(1)
    print(f"✅ Hoàn thành: {desc}")


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline ASQP")
    parser.add_argument("--step", choices=[
        "prepare", "train_bilstm", "train_phobert", "train_vit5",
        "compare", "demo"
    ], help="Bước cần chạy")
    parser.add_argument("--all", action="store_true",
                        help="Chạy tất cả các bước (trừ demo)")
    args = parser.parse_args()

    steps = {
        "prepare":       (f"{PYTHON} src/training/prepare_data.py",
                          "Chuẩn bị dữ liệu (prepare_data)"),
        "train_bilstm":  (f"{PYTHON} src/training/train_bilstm.py",
                          "Train BiLSTM+CRF baseline"),
        "train_phobert": (f"{PYTHON} src/training/train_phobert.py",
                          "Train PhoBERT fine-tune"),
        "train_vit5":    (f"{PYTHON} src/training/train_vit5.py",
                          "Train ViT5 seq2seq"),
        "compare":       (f"{PYTHON} src/training/compare_models.py",
                          "So sánh kết quả 3 models"),
        "demo":          (f"streamlit run src/deployment/app.py",
                          "Chạy Web Demo (Streamlit)"),
    }

    if args.all:
        order = ["prepare", "train_bilstm", "train_phobert", "train_vit5", "compare"]
        for step in order:
            cmd, desc = steps[step]
            run(cmd, desc)
        print("\n🎉 Pipeline hoàn tất! Chạy demo bằng:")
        print("   streamlit run src/deployment/app.py")
    elif args.step:
        cmd, desc = steps[args.step]
        run(cmd, desc)
    else:
        parser.print_help()
        print("\n💡 Gợi ý: bắt đầu bằng --step prepare")


if __name__ == "__main__":
    main()
