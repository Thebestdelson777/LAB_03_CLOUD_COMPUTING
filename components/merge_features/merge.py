import argparse
import os
import pandas as pd


def read_parquet(folder_path: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(folder_path, "data.parquet"))


def write_parquet(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    base_df = read_parquet(args.base)
    len_df = read_parquet(args.length)
    sent_df = read_parquet(args.sentiment)
    tfidf_df = read_parquet(args.tfidf)
    sbert_df = read_parquet(args.sbert)

    keys = ["asin", "reviewerID"]

    base_df = base_df[keys + ["overall"]].drop_duplicates(subset=keys)
    len_df = len_df.drop_duplicates(subset=keys)
    sent_df = sent_df.drop_duplicates(subset=keys)
    tfidf_df = tfidf_df.drop_duplicates(subset=keys)
    sbert_df = sbert_df.drop_duplicates(subset=keys)

    merged = base_df.merge(len_df, on=keys, how="inner")
    merged = merged.merge(sent_df, on=keys, how="inner")
    merged = merged.merge(tfidf_df, on=keys, how="inner")
    merged = merged.merge(sbert_df, on=keys, how="inner")

    merged = merged.drop_duplicates(subset=keys)

    write_parquet(merged, args.out)
    print("Merged shape:", merged.shape)
    print("Columns:", len(merged.columns))


if __name__ == "__main__":
    main()
