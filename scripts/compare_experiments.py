"""
Compare evaluation outputs from baseline, PPO+MLP, and PPO+KAN runs.

Example:
python scripts/compare_experiments.py \
  --labels sft ppo_mlp ppo_kan \
  --files results/kaggle_finetuned_results.json results/ppo_mlp_eval.json results/ppo_kan_eval.json \
  --out results/experiment_comparison.csv
"""

import json
import argparse
from pathlib import Path

import pandas as pd


def load_eval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "compilable_rate": payload.get("compilable_rate", 0.0),
        "avg_pass_rate": payload.get("avg_pass_rate", 0.0),
        "avg_line_coverage": payload.get("avg_line_coverage", 0.0),
        "avg_branch_coverage": payload.get("avg_branch_coverage", 0.0),
        "avg_reward": payload.get("avg_reward", 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--out", default="results/experiment_comparison.csv")
    args = parser.parse_args()

    if len(args.labels) != len(args.files):
        raise ValueError("--labels and --files must have the same length")

    rows = []
    for label, file_path in zip(args.labels, args.files):
        metrics = load_eval(file_path)
        rows.append({"experiment": label, **metrics, "source_file": file_path})

    df = pd.DataFrame(rows)

    # Percentage-friendly rounding
    for col in ["compilable_rate", "avg_pass_rate", "avg_line_coverage", "avg_branch_coverage"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    if "avg_reward" in df.columns:
        df["avg_reward"] = df["avg_reward"].round(4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("=== Experiment Comparison ===")
    print(df.to_string(index=False))
    print(f"\nSaved comparison CSV to {out_path}")


if __name__ == "__main__":
    main()
