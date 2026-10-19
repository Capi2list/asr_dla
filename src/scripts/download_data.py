import argparse
import gzip
import shutil
from pathlib import Path

import wget
from huggingface_hub import hf_hub_download

URL = {"LM": "https://www.openslr.org/resources/11/4-gram.arpa.gz"}


def download_checkpoint(dir="data/"):
    Path(dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = hf_hub_download(
        repo_id="capibucca/asr_checkpoint",
        filename="best_model.pth",
        local_dir=dir,
    )
    return checkpoint_path


def download_LM(dir="data/"):
    Path(dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = hf_hub_download(
        repo_id="capibucca/asr_checkpoint",
        filename="best_model.pth",
        local_dir=dir,
    )
    return checkpoint_path


def download_all(dir="data/"):
    checkpoint_path = download_checkpoint(dir)
    lm_path = download_LM(dir)
    return {"lm_path": lm_path, "checkpoint_path": checkpoint_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/")
    args = parser.parse_args()
    download_all(args.output_dir)
