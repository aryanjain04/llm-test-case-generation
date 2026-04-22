"""
Build candidate-reward dataset for critic/reranker training.

Pipeline:
1. Sample unique functions from JSONL function-test dataset.
2. Generate K test candidates per function with fine-tuned CodeT5.
3. Execute each candidate in sandbox to obtain reward/pass/coverage.
4. Extract static feature vector (function + candidate features).
5. Save rows for supervised critic training.

Example:
python scripts/build_critic_dataset.py \
  --input datasets/train_combined.jsonl \
  --checkpoint checkpoints/codet5-finetuned/best \
  --base-model Salesforce/codet5-base \
  --max-functions 300 --num-candidates 5 \
  --output datasets/critic_reranker_train.jsonl
"""

from __future__ import annotations

import sys
import json
import random
import argparse
from pathlib import Path

import jsonlines

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.codet5_generator import CodeT5Generator, PROMPT_TEMPLATE_FINETUNE
from src.execution.sandbox import TestExecutor
from src.rl.reranker_features import build_combined_feature_vector


def load_unique_functions(path: str, max_functions: int, seed: int) -> list[dict]:
    rows = []
    seen = set()

    with jsonlines.open(path) as reader:
        for row in reader:
            code = row.get("function_code", "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            rows.append(
                {
                    "function_name": row.get("function_name", "unknown"),
                    "function_code": code,
                    "metadata": row.get("metadata", {}),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)

    if max_functions > 0:
        rows = rows[:max_functions]

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/train_combined.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/codet5-finetuned/best")
    parser.add_argument("--base-model", default="Salesforce/codet5-base")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-functions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="datasets/critic_reranker_train.jsonl")
    parser.add_argument("--meta-output", default="datasets/critic_reranker_train_meta.json")
    args = parser.parse_args()

    random.seed(args.seed)

    pool = load_unique_functions(args.input, max_functions=args.max_functions, seed=args.seed)
    if not pool:
        raise ValueError(f"No functions found in {args.input}")

    print(f"Loaded {len(pool)} unique functions from {args.input}")

    generator = CodeT5Generator.from_checkpoint(args.checkpoint, base_model=args.base_model)
    executor = TestExecutor(timeout=30)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    compilable_rows = 0
    feature_names = None

    with jsonlines.open(out_path, mode="w") as writer:
        for i, item in enumerate(pool, start=1):
            fname = item["function_name"]
            fcode = item["function_code"]

            print(f"[{i}/{len(pool)}] {fname}")
            candidates = generator.generate(
                fcode,
                prompt_template=PROMPT_TEMPLATE_FINETUNE,
                num_return_sequences=args.num_candidates,
                temperature=args.temperature,
                do_sample=True,
            )

            for j, cand in enumerate(candidates, start=1):
                exec_result = executor.execute(fcode, cand)
                vec, names = build_combined_feature_vector(fcode, cand)

                if feature_names is None:
                    feature_names = names

                row = {
                    "function_name": fname,
                    "candidate_id": j,
                    "features": vec,
                    "target_reward": exec_result.reward,
                    "target_pass_rate": exec_result.pass_rate,
                    "target_line_coverage": exec_result.line_coverage,
                    "target_branch_coverage": exec_result.branch_coverage,
                    "compilable": exec_result.compilable,
                    "total_tests": exec_result.total,
                }
                writer.write(row)

                total_rows += 1
                compilable_rows += int(exec_result.compilable)

    meta = {
        "input": args.input,
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "num_functions": len(pool),
        "num_candidates": args.num_candidates,
        "total_rows": total_rows,
        "compilable_rows": compilable_rows,
        "compilable_rate": (compilable_rows / total_rows) if total_rows else 0.0,
        "feature_names": feature_names or [],
        "target": "target_reward",
    }

    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Done ===")
    print(f"Dataset rows: {total_rows}")
    print(f"Compilable rows: {compilable_rows} ({meta['compilable_rate']:.1%})")
    print(f"Saved: {out_path}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
