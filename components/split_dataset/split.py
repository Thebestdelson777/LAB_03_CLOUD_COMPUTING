import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def read_uri_folder(path: str) -> pd.DataFrame:
    parquet_files = glob.glob(os.path.join(path, "*.parquet"))
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
    if parquet_files:
        return pd.read_parquet(parquet_files[0])

    raise FileNotFoundError(f"No parquet file found in: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--deploy_ratio", type=float, default=0.10)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio + args.deploy_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    df = read_uri_folder(args.data)
    df = df.reset_index(drop=True)
    df["record_id"] = df.index.astype(str)
    # Optional time-aware deployment split
    if "review_year" in df.columns:
        df = df.sort_values("review_year").reset_index(drop=True)

        deploy_size = int(len(df) * args.deploy_ratio)
        if deploy_size < 1:
            raise ValueError("Deployment split is empty. Increase dataset size.")

        df_non_deploy = df.iloc[:-deploy_size].copy()
        deploy_df = df.iloc[-deploy_size:].copy()

        remaining = 1.0 - args.deploy_ratio
        train_share = args.train_ratio / remaining
        val_share = args.val_ratio / remaining
        test_share = args.test_ratio / remaining

        train_df, temp_df = train_test_split(
            df_non_deploy,
            test_size=(1 - train_share),
            random_state=args.seed,
            shuffle=True
        )

        val_adjusted = val_share / (val_share + test_share)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_adjusted),
            random_state=args.seed,
            shuffle=True
        )
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - args.train_ratio),
            random_state=args.seed,
            shuffle=True
        )

        remaining_ratio = args.val_ratio + args.test_ratio + args.deploy_ratio
        val_adjusted = args.val_ratio / remaining_ratio
        test_adjusted = args.test_ratio / remaining_ratio

        val_df, temp_df2 = train_test_split(
            temp_df,
            test_size=(1 - val_adjusted),
            random_state=args.seed,
            shuffle=True
        )

        deploy_adjusted = args.deploy_ratio / (args.test_ratio + args.deploy_ratio)
        test_df, deploy_df = train_test_split(
            temp_df2,
            test_size=deploy_adjusted,
            random_state=args.seed,
            shuffle=True
        )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)
    os.makedirs(args.deploy_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)
    deploy_df.to_parquet(os.path.join(args.deploy_out, "data.parquet"), index=False)

    print("Split complete.")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))
    print("Deploy:", len(deploy_df))


if __name__ == "__main__":
    main()
