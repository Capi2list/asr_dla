import argparse
import json
from pathlib import Path
from src.metrics.utils import calc_cer, calc_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    N = 0
    cer = 0.
    wer = 0.
    for cur_file in input_dir.iterdir():
        if cur_file.is_file() and cur_file.suffix == ".json":
            with open(cur_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "prediction" not in data.keys() or "target" not in data.keys():
                    continue
                wer += calc_wer(data["target"], data["prediction"])
                cer += calc_cer(data["target"], data["prediction"])
                N += 1
    if N == 0:
        print("Have not found any output files.")
    else:
        print(f"CER: {cer/N}\nWER: {wer/N}")


if __name__ == "__main__":
    main()