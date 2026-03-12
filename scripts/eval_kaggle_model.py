"""
Load a Kaggle-trained checkpoint and evaluate on functions.py

Usage:
    1. Download the checkpoint from Kaggle Output tab
    2. Place it in checkpoints/codet5-base-finetuned/best/
       (or checkpoints/codet5-base-finetuned/merged/)
    3. Run: python scripts/eval_kaggle_model.py

This handles both LoRA adapter checkpoints and fully merged models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.ast_analysis.feature_extractor import extract_functions_from_source, extract_features_from_source
from src.model.codet5_generator import CodeT5Generator, PROMPT_TEMPLATE_FINETUNE
from src.execution.sandbox import TestExecutor
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/codet5-base-finetuned/best",
        help="Path to Kaggle checkpoint (LoRA adapter or merged model)",
    )
    parser.add_argument(
        "--base-model",
        default="Salesforce/codet5-base",
        help="Base model (must match what was used for training)",
    )
    parser.add_argument("--functions-file", default="functions.py")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="results/kaggle_finetuned_results.json")
    args = parser.parse_args()

    # Check checkpoint exists
    cp = Path(args.checkpoint)
    if not cp.exists():
        print(f"ERROR: Checkpoint not found at {cp}")
        print(f"\nExpected one of:")
        print(f"  {cp / 'adapter_config.json'}  (LoRA adapter)")
        print(f"  {cp / 'config.json'}          (merged model)")
        print(f"\nDownload from Kaggle and place files in: {cp}")
        return

    # Load functions
    with open(args.functions_file, "r", encoding="utf-8") as f:
        source = f.read()

    functions = extract_functions_from_source(source)
    features = extract_features_from_source(source)

    print(f"Loaded {len(functions)} functions")
    print(f"\n=== Code Features ===")
    for feat in features:
        print(f"  {feat.name}: CC={feat.cyclomatic_complexity}, LOC={feat.loc}")

    # Load model — detect if LoRA or merged
    is_lora = (cp / "adapter_config.json").exists()
    if is_lora:
        print(f"\nLoading LoRA checkpoint from {cp}")
        generator = CodeT5Generator.from_checkpoint(
            str(cp), base_model=args.base_model
        )
    else:
        print(f"\nLoading merged checkpoint from {cp}")
        generator = CodeT5Generator(model_name_or_path=str(cp))

    # Generate and evaluate
    print(f"\n=== Generating Tests ({args.num_candidates} candidates each) ===")
    generated_tests = []
    executor = TestExecutor()

    for func_name, func_source in functions:
        print(f"\n--- {func_name} ---")

        candidates = generator.generate(
            func_source,
            prompt_template=PROMPT_TEMPLATE_FINETUNE,
            num_return_sequences=args.num_candidates,
            temperature=args.temperature,
        )

        best_test = None
        best_reward = -float("inf")

        for i, test_code in enumerate(candidates):
            result = executor.execute(func_source, test_code)
            status = "✓" if result.pass_rate > 0 else "✗"
            print(f"  [{status}] Candidate {i+1}: pass={result.pass_rate:.0%}, "
                  f"cov={result.line_coverage:.0f}%, reward={result.reward:.3f}")

            if result.reward > best_reward:
                best_reward = result.reward
                best_test = test_code

        generated_tests.append((func_name, best_test or (candidates[0] if candidates else "")))
        print(f"  → Best reward: {best_reward:.3f}")

    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    evaluator = Evaluator()
    result = evaluator.evaluate(functions=functions, generated_tests=generated_tests)
    print(result.summary())
    evaluator.save_results(result, args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
