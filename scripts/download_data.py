"""Download and preprocess XL-Sum Hindi into data/processed/*.jsonl.

Run this once before training or evaluating anything:
    python scripts/download_data.py
"""

from textsumm.data import download_and_clean

if __name__ == "__main__":
    download_and_clean()
