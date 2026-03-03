import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "data.parquet"))

    text_col = None
    for c in ["reviewText", "text", "normalized_text", "review_text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True)

    emb_cols = [f"sbert_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    keep_cols = [c for c in ["asin", "reviewerID"] if c in df.columns]
    out_df = pd.concat([df[keep_cols].reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("SBERT output shape:", out_df.shape)

if __name__ == "__main__":
    main()
