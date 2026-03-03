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
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)

    # optional args kept for compatibility, but not used in the “lite” merge
    parser.add_argument("--tfidf", type=str, required=False)
    parser.add_argument("--sbert", type=str, required=False)

    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df_len = read_parquet(args.length)
    df_sent = read_parquet(args.sentiment)

    keys = [k for k in ["asin", "reviewerID"] if k in df_len.columns and k in df_sent.columns]
    if not keys:
        raise ValueError(
            "No join keys found in BOTH datasets. "
            f"length cols={list(df_len.columns)} sentiment cols={list(df_sent.columns)}"
        )

    merged = df_len.merge(df_sent, on=keys, how="inner")

    write_parquet(merged, args.out)
    print("✅ Merged (length + sentiment). Shape:", merged.shape)


if __name__ == "__main__":
    main()
