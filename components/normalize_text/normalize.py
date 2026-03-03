import argparse
import os
import re
import string
import glob
import pandas as pd


def read_uri_folder(path: str) -> pd.DataFrame:
    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    raise FileNotFoundError(f"No parquet file found in: {path}")


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = read_uri_folder(args.data)

    # expected column name in this lab dataset
    if "reviewText" not in df.columns:
        raise ValueError(f"Expected column 'reviewText' not found. Columns: {list(df.columns)}")

    df["reviewText"] = df["reviewText"].apply(normalize_text)

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Normalization complete:", len(df))


if __name__ == "__main__":
    main()
