import json
import os
import joblib
import pandas as pd

model = None
feature_columns = None


def init():
    global model, feature_columns

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set.")

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        # fallback: search recursively in case Azure places it inside a subfolder
        for root, _, files in os.walk(model_dir):
            if "model.pkl" in files:
                model_path = os.path.join(root, "model.pkl")
                break

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found under: {model_dir}")

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
    else:
        raise RuntimeError("Saved model artifact is not in the expected dictionary format.")


def run(raw_data):
    try:
        data = json.loads(raw_data)

        if "data" not in data:
            return {"error": "Request body must contain a 'data' field."}

        rows = data["data"]

        if not isinstance(rows, list):
            return {"error": "'data' must be a list."}

        # Case 1: list of dictionaries with column names
        if len(rows) > 0 and isinstance(rows[0], dict):
            df = pd.DataFrame(rows)

            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0.0

            X = df[feature_columns].copy()

        # Case 2: list of lists already matching feature order
        elif len(rows) > 0 and isinstance(rows[0], list):
            X = pd.DataFrame(rows, columns=feature_columns)

        # Case 3: empty input
        else:
            X = pd.DataFrame(columns=feature_columns)

        X = X.fillna(0.0).astype("float32")

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        return {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
