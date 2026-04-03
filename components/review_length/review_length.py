import argparse
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = read_uri_folder(args.data)

    if "reviewText" not in df.columns:
        raise ValueError(f"Expected column 'reviewText' not found. Columns: {list(df.columns)}")

    text = df["reviewText"].fillna("").astype(str)

    out_df = pd.DataFrame()
    out_df["record_id"] = df["record_id"]
    out_df["asin"] = df["asin"]
    out_df["reviewerID"] = df["reviewerID"]

    out_df["review_length_chars"] = text.str.len()
    out_df["review_length_words"] = text.str.split().apply(len)

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Review length features complete:", out_df.shape)


if __name__ == "__main__":
    main()
