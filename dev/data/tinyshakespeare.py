"""
Download and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken.
- output: a newly created tinyshakespeare/ folder.
"""
import os
from data_common import download_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping downloading...")

def tokenize():


if __name__ == "__main__":
    download()
    tokenize()