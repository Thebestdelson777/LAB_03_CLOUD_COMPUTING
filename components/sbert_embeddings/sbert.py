import argparse
import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer


def read_uri_folder(path: str) -> pd.DataFrame:
    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    raise FileNotFoundError(f"No parquet file found in: {path}")


def write_parquet(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)


def get_text_column(df: pd.DataFrame) -> str:
    for col in ["reviewText", "review_text", "text"]:
        if col in df.columns:
            return col
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")


def shorten_text(text, max_chars=500):
    text = "" if pd.isna(text) else str(text)
    return text[:max_chars]


def build_output_df(df: pd.DataFrame, emb, dims: int):
    emb = emb[:, :dims]
    emb_cols = [f"sbert_{i}" for i in range(dims)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    out_df = pd.concat(
        [
            df[["record_id", "asin", "reviewerID"]].reset_index(drop=True),
            emb_df.reset_index(drop=True)
        ],
        axis=1
    )

    return out_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--deploy", type=str, required=True)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    args = parser.parse_args()

    train_df = read_uri_folder(args.train)
    val_df = read_uri_folder(args.val)
    test_df = read_uri_folder(args.test)
    deploy_df = read_uri_folder(args.deploy)

    text_col = get_text_column(train_df)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_texts = train_df[text_col].fillna("").astype(str).apply(shorten_text).tolist()
    val_texts = val_df[text_col].fillna("").astype(str).apply(shorten_text).tolist()
    test_texts = test_df[text_col].fillna("").astype(str).apply(shorten_text).tolist()
    deploy_texts = deploy_df[text_col].fillna("").astype(str).apply(shorten_text).tolist()

    train_emb = model.encode(train_texts, batch_size=32, show_progress_bar=True)
    val_emb = model.encode(val_texts, batch_size=32, show_progress_bar=True)
    test_emb = model.encode(test_texts, batch_size=32, show_progress_bar=True)
    deploy_emb = model.encode(deploy_texts, batch_size=32, show_progress_bar=True)

    train_out = build_output_df(train_df, train_emb, args.dims)
    val_out = build_output_df(val_df, val_emb, args.dims)
    test_out = build_output_df(test_df, test_emb, args.dims)
    deploy_out = build_output_df(deploy_df, deploy_emb, args.dims)

    write_parquet(train_out, args.train_out)
    write_parquet(val_out, args.val_out)
    write_parquet(test_out, args.test_out)
    write_parquet(deploy_out, args.deploy_out)

    print("SBERT complete")
    print("Train shape:", train_out.shape)
    print("Val shape:", val_out.shape)
    print("Test shape:", test_out.shape)
    print("Deploy shape:", deploy_out.shape)


if __name__ == "__main__":
    main()
