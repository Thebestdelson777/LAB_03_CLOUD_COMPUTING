import argparse
import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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


def build_output_df(keys_df: pd.DataFrame, matrix, feature_names):
    tfidf_df = pd.DataFrame(
        matrix.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names]
    )

    out_df = pd.concat(
        [
            keys_df[["record_id", "asin", "reviewerID"]].reset_index(drop=True),
            tfidf_df.reset_index(drop=True)
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

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500
    )

    X_train = vectorizer.fit_transform(train_df[text_col].fillna("").astype(str))
    X_val = vectorizer.transform(val_df[text_col].fillna("").astype(str))
    X_test = vectorizer.transform(test_df[text_col].fillna("").astype(str))
    X_deploy = vectorizer.transform(deploy_df[text_col].fillna("").astype(str))

    feature_names = vectorizer.get_feature_names_out()

    train_out = build_output_df(train_df, X_train, feature_names)
    val_out = build_output_df(val_df, X_val, feature_names)
    test_out = build_output_df(test_df, X_test, feature_names)
    deploy_out = build_output_df(deploy_df, X_deploy, feature_names)

    write_parquet(train_out, args.train_out)
    write_parquet(val_out, args.val_out)
    write_parquet(test_out, args.test_out)
    write_parquet(deploy_out, args.deploy_out)

    print("TF-IDF complete")
    print("Train shape:", train_out.shape)
    print("Val shape:", val_out.shape)
    print("Test shape:", test_out.shape)
    print("Deploy shape:", deploy_out.shape)


if __name__ == "__main__":
    main()
