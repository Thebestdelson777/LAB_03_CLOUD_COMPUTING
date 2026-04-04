import json
import math
import subprocess
import tempfile
import pandas as pd
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Endpoint details
# --------------------------------------------------
ENDPOINT_URL = "https://amazon-review-endpoint-60302101.qatarcentral.inference.ml.azure.com/score"
API_KEY = "<api-key-git-wont-let-me-push-key-present-in-readme>"

DEPLOY_DATA_PATH = "./named-outputs/exported_data/data.parquet"
BATCH_SIZE = 100  # safe size for endpoint requests


def load_data(path):
    df = pd.read_parquet(path)
    print(f"Loaded deployment data: {df.shape}")
    return df


def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")

    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


def score_batch(batch_values):
    payload = {
        "data": batch_values
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_file = f.name

    cmd = [
        "curl",
        "-k",
        "-s",
        "-X", "POST",
        ENDPOINT_URL,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {API_KEY}",
        "--data-binary", f"@{payload_file}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")

    response_text = result.stdout

    if response_text.strip().startswith("<html>"):
        raise RuntimeError(f"Endpoint returned HTML instead of JSON:\n{response_text[:500]}")

    response_json = json.loads(response_text)

    if "error" in response_json:
        raise RuntimeError(response_json["error"])

    return response_json["predictions"]


def main():
    df = load_data(DEPLOY_DATA_PATH)
    df = create_labels(df)

    # best final feature configuration = sbert + tfidf
    tfidf_cols = sorted([col for col in df.columns if col.startswith("tfidf_")])
    sbert_cols = sorted(
        [col for col in df.columns if col.startswith("sbert_")],
        key=lambda x: int(x.split("_")[1])
    )

    feature_columns = sbert_cols + tfidf_cols

    X = df[feature_columns].copy()
    X = X.fillna(0.0).astype("float32")
    y_true = df["label"].tolist()

    X_values = X.values.tolist()

    all_preds = []
    total_batches = math.ceil(len(X_values) / BATCH_SIZE)

    for i in range(total_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(X_values))
        batch_values = X_values[start:end]

        print(f"Scoring batch {i + 1}/{total_batches} rows {start}:{end}")
        batch_preds = score_batch(batch_values)
        all_preds.extend(batch_preds)

    acc = accuracy_score(y_true, all_preds)
    print("Deployment accuracy:", acc)
    print("First 20 predictions:", all_preds[:20])


if __name__ == "__main__":
    main()
