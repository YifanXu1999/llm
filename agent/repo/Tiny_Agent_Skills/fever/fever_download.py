import argparse
import os
from urllib.request import Request, urlopen


URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
    "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description="Download FEVER dataset files.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--splits", nargs="+", default=["paper_dev", "paper_test", "train"])
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    for split in args.splits:
        url = URLS.get(split)
        if not url:
            raise ValueError(f"Unknown split: {split}")
        filename = os.path.basename(url)
        path = os.path.join(args.data_dir, filename)
        if os.path.exists(path):
            print(f"Exists: {path}")
            continue
        print(f"Downloading {url} -> {path}")
        req = Request(url, headers={"User-Agent": "Tiny_Agent_Skills/0.1 (local)"})
        with urlopen(req) as resp, open(path, "wb") as f:
            f.write(resp.read())


if __name__ == "__main__":
    main()
