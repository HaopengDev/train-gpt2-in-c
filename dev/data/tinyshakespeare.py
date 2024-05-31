"""
Download and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken.
- output: a newly created tinyshakespeare/ folder.
"""
import os
import tiktoken
from data_common import download_file, write_datafile

DATA_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})

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
    data_filename = os.path.join(DATA_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    # treat every statement in the dialog as a separate doc.
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    tokens = encode(text)
    validation_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    validation_filename = os.path.join(DATA_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_DIR, "tiny_shakespeare_train.bin")
    write_datafile(validation_filename, validation_tokens)
    write_datafile(train_filename, train_tokens)

if __name__ == "__main__":
    download()
    tokenize()