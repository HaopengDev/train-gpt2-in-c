import requests
from tqdm import tqdm
import numpy as np

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm (
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def write_datafile(filename, tokens):
    assert len(tokens) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240521 # magic
    header[1] = 1 # version
    header[2] = len(tokens)
    if not isinstance(tokens, np.ndarray) or not tokens.dtype == np.uint16:
        maxtoken = 2**16
        assert all(0 <= t < maxtoken for t in tokens)
        tokens_np = np.array(tokens, dtype=np.uint16)
    else:
        tokens_np = tokens

    print(f"writing {len(tokens):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())