import argparse
from pathlib import Path
import pandas as pd


def prepare_data(imgtools_path: str | Path, output_path: str | Path):

    imgtools_path = Path(imgtools_path)

    index_csv = pd.read_csv(imgtools_path / f"{imgtools_path.name}_index-simple.csv")

    # DO THE NPZ CONVERSION HERE


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--imgtools_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    args = argparser.parse_args()

    prepare_data(args.imgtools_path, args.output_path)