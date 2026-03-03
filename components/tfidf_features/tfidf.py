import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, required=True)
    p.add_argument("--val", type=str, required=True)
    p.add_argument("--test", type=str, required=True)
    p.add_argument("--train_out", type=str, required=True)
    p.add_argument("--val_out", type=str, required=True)
    p.add_argument("--test_out", type=str, required=True)
    return p.parse_args()


def read_parquet(folder_path: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(folder_path, "data.parquet"))


def write_parquet(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)


def get_text_column(df: pd.DataFrame) -> str:
    for col in ["reviewText", "review_text", "text"]:
        if col in df.columns:
            return col
    raise ValueError(f"No text column found. Columns: {list(df.columns)}")


def sparse_to_long(X, feature_names, keys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CSR matrix to long format:
    row_id, term, tfidf (+ optional keys like reviewerID/asin)
    """
    X = X.tocsr()
    coo = X.tocoo()

    terms = np.asarray(feature_names, dtype=object)[coo.col]

    out = pd.DataFrame({
        "row_id": coo.row.astype(np.int32),
        "term": terms,
        "tfidf": coo.data.astype(np.float32),
    })

    for key in ["reviewerID", "asin"]:
        if key in keys_df.columns:
            out[key] = keys_df[key].iloc[out["row_id"]].values

    cols = [c for c in ["reviewerID", "asin", "row_id", "term", "tfidf"] if c in out.columns]
    return out[cols]


def main():
    args = parse_args()

    train_df = read_parquet(args.train)
    val_df = read_parquet(args.val)
    test_df = read_parquet(args.test)

    text_col = get_text_column(train_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        dtype=np.float32
    )

    train_text = train_df[text_col].fillna("").astype(str)
    vectorizer.fit(train_text)

    feature_names = vectorizer.get_feature_names_out()

    X_train = vectorizer.transform(train_text)
    X_val = vectorizer.transform(val_df[text_col].fillna("").astype(str))
    X_test = vectorizer.transform(test_df[text_col].fillna("").astype(str))

    train_out = sparse_to_long(X_train, feature_names, train_df)
    val_out = sparse_to_long(X_val, feature_names, val_df)
    test_out = sparse_to_long(X_test, feature_names, test_df)

    write_parquet(train_out, args.train_out)
    write_parquet(val_out, args.val_out)
    write_parquet(test_out, args.test_out)

    print("TFIDF written in sparse long format.")
    print("Train nonzeros:", len(train_out))
    print("Val nonzeros:", len(val_out))
    print("Test nonzeros:", len(test_out))


if __name__ == "__main__":
    main()
