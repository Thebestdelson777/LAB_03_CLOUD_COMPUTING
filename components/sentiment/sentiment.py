import argparse
import os
import glob
import pandas as pd

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_uri_folder(path: str) -> pd.DataFrame:
    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    raise FileNotFoundError(f"No parquet found under: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = read_uri_folder(args.data)

    if "reviewText" not in df.columns:
        raise ValueError("Expected column 'reviewText' in input data")

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    scores = df["reviewText"].fillna("").astype(str).apply(sia.polarity_scores)
    df["sentiment_pos"] = scores.apply(lambda d: d["pos"])
    df["sentiment_neg"] = scores.apply(lambda d: d["neg"])
    df["sentiment_neu"] = scores.apply(lambda d: d["neu"])
    df["sentiment_compound"] = scores.apply(lambda d: d["compound"])

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("Output rows:", len(df))


if __name__ == "__main__":
    main()
