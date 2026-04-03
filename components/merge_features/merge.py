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

    keys = ["record_id"]

    base_df = base_df[["record_id", "asin", "reviewerID", "overall"]]
    len_df = len_df
    sent_df = sent_df
    tfidf_df = tfidf_df
    sbert_df = sbert_df

    merged = base_df.merge(len_df.drop(columns=["asin", "reviewerID"]), on=keys, how="left")
    merged = merged.merge(sent_df.drop(columns=["asin", "reviewerID"]), on=keys, how="left")
    merged = merged.merge(tfidf_df.drop(columns=["asin", "reviewerID"]), on=keys, how="left")
    merged = merged.merge(sbert_df.drop(columns=["asin", "reviewerID"]), on=keys, how="left")
    merged = merged.drop_duplicates(subset=keys)

    write_parquet(merged, args.out)
    print("Merged shape:", merged.shape)
    print("Columns:", len(merged.columns))


if __name__ == "__main__":
    main()
