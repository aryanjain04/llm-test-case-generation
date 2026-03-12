"""
Run the zero-shot and fine-tuned baselines on functions.py

Usage:
    python scripts/run_baseline.py                        # zero-shot only
    python scripts/run_baseline.py --checkpoint checkpoints/codet5-finetuned/best  # fine-tuned
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ast_analysis.feature_extractor import (
    extract_features_from_source,
    extract_functions_from_source,
)
from src.model.codet5_generator import (
    CodeT5Generator,
    PROMPT_TEMPLATE_V1,
    PROMPT_TEMPLATE_FINETUNE,
)
from src.execution.sandbox import TestExecutor
from src.evaluation.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-file", default="functions.py")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", default="results/baseline_results.json")
    args = parser.parse_args()

    # 1. Load functions
    with open(args.functions_file, "r", encoding="utf-8") as f:
        source = f.read()

    functions = extract_functions_from_source(source)
    print(f"Loaded {len(functions)} functions from {args.functions_file}\n")

    # 2. Extract features
    features = extract_features_from_source(source)
    print("=== Code Features ===")
    for feat in features:
        print(f"  {feat.name}: CC={feat.cyclomatic_complexity}, LOC={feat.loc}, "
              f"params={feat.param_count}, branches={feat.branch_count}")
    print()

    # 3. Load model
    if args.checkpoint:
        print(f"Loading fine-tuned model from {args.checkpoint}")
        generator = CodeT5Generator.from_checkpoint(args.checkpoint)
        prompt_template = PROMPT_TEMPLATE_FINETUNE  # Must match training prompt
    else:
        print("Using zero-shot CodeT5-small")
        generator = CodeT5Generator()
        prompt_template = PROMPT_TEMPLATE_V1

    # 4. Generate tests
    print("\n=== Generating Tests ===")
    generated_tests = []
    executor = TestExecutor()

    for func_name, func_source in functions:
        print(f"\n--- {func_name} ---")

        candidates = generator.generate(
            func_source,
            prompt_template=prompt_template,
            num_return_sequences=args.num_candidates,
            temperature=args.temperature,
        )

        # Pick the best candidate by execution
        best_test = None
        best_reward = -float("inf")

        for i, test_code in enumerate(candidates):
            result = executor.execute(func_source, test_code)
            print(f"  Candidate {i+1}: pass_rate={result.pass_rate:.0%}, "
                  f"coverage={result.line_coverage:.0f}%, reward={result.reward:.3f}")

            if result.reward > best_reward:
                best_reward = result.reward
                best_test = test_code

        if best_test:
            generated_tests.append((func_name, best_test))
            print(f"  → Selected candidate with reward={best_reward:.3f}")
        else:
            generated_tests.append((func_name, candidates[0] if candidates else ""))

    # 5. Evaluate
    print("\n=== Evaluation ===")

    # Load reference tests for BLEU comparison
    ref_test_file = Path("test_llm-generated.py")
    reference_tests = None
    if ref_test_file.exists():
        ref_source = ref_test_file.read_text(encoding="utf-8")
        # For BLEU, we'd need per-function reference tests
        # Simplified: skip BLEU for now in baseline
        print("Reference tests found (BLEU comparison available)")

    evaluator = Evaluator()
    result = evaluator.evaluate(
        functions=functions,
        generated_tests=generated_tests,
    )

    print(result.summary())

    # 6. Save results
    evaluator.save_results(result, args.output)
    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
