import argparse
import os
import time

import azureml.mlflow
import joblib
import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--c_value", type=float, default=0.89)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument(
        "--feature_set",
        type=str,
        default="all",
        choices=["sbert", "sbert_tfidf", "all"]
    )

    return parser.parse_args()


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    df = pd.read_parquet(path)
    print(f"Loaded {path} -> shape: {df.shape}")
    return df


def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")

    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


def get_feature_columns(df, feature_set):
    length_cols = ["review_length_chars", "review_length_words"]
    sentiment_cols = [
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_compound",
    ]
    tfidf_cols = sorted([col for col in df.columns if col.startswith("tfidf_")])
    sbert_cols = sorted(
        [col for col in df.columns if col.startswith("sbert_")],
        key=lambda x: int(x.split("_")[1])
    )

    if feature_set == "sbert":
        feature_cols = sbert_cols
    elif feature_set == "sbert_tfidf":
        feature_cols = sbert_cols + tfidf_cols
    elif feature_set == "all":
        feature_cols = length_cols + sentiment_cols + tfidf_cols + sbert_cols
    else:
        raise RuntimeError(f"Unsupported feature_set: {feature_set}")

    feature_cols = [col for col in feature_cols if col in df.columns]

    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns were found in the dataset.")

    return feature_cols


def build_features(df, feature_cols):
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing feature columns: {missing_cols[:10]}")

    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0.0).astype("float32")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise RuntimeError("Feature matrix is empty.")

    return X


def evaluate(model, X, y, split):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, probs)

    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_precision", precision)
    mlflow.log_metric(f"{split}_recall", recall)
    mlflow.log_metric(f"{split}_f1", f1)
    mlflow.log_metric(f"{split}_auc", auc)

    print(f"\n{split.upper()} METRICS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")


def main():
    args = parse_args()
    start_time = time.time()

    mlflow.start_run()

    print("Loading train/val/test data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("Creating binary labels from overall rating...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Selecting feature columns...")
    feature_cols = get_feature_columns(train_df, args.feature_set)

    print(f"Feature set: {args.feature_set}")
    print(f"Total feature columns found: {len(feature_cols)}")
    print(f"First 10 feature columns: {feature_cols[:10]}")

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("feature_set", args.feature_set)
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("label_rule", "overall >= 4")
    mlflow.log_param("solver", "saga")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("c_value", args.c_value)
    mlflow.log_param("max_iter", args.max_iter)

    print("Building feature matrices...")
    X_train = build_features(train_df, feature_cols)
    X_val = build_features(val_df, feature_cols)
    X_test = build_features(test_df, feature_cols)

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        C=args.c_value,
        solver="saga",
        max_iter=args.max_iter,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model artifact...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")

    artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_set": args.feature_set,
    }
    joblib.dump(artifact, model_path)

    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)

    print(f"\nTraining runtime (seconds): {runtime:.2f}")
    print(f"Saved model to: {model_path}")
    print("Done.")

    mlflow.end_run()


if __name__ == "__main__":
    main()
