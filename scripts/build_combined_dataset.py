"""
Build a high-quality training dataset by combining:
1. Our mined standalone examples (cleaned, filtered)
2. MBPP (Mostly Basic Python Problems) - 974 clean function+test pairs
3. HumanEval-like simple function+test pairs

This gives us a proper training set for CodeT5 fine-tuning.
"""
import json
import re
import ast
import textwrap
from pathlib import Path

try:
    import jsonlines
except ImportError:
    raise ImportError("pip install jsonlines")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("pip install datasets")


def load_mbpp():
    """
    Load MBPP dataset from HuggingFace.
    MBPP has: task_id, text (description), code (solution), test_list (list of assert strings)
    """
    print("Loading MBPP from HuggingFace...")
    ds = load_dataset("mbpp", "full", trust_remote_code=True)

    examples = []
    skipped = 0

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            code = item["code"].strip()
            test_list = item.get("test_list", [])
            challenge_test_list = item.get("challenge_test_list", [])

            if not code or not test_list:
                skipped += 1
                continue

            # Verify function code parses
            try:
                tree = ast.parse(code)
                func_names = [
                    n.name for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef)
                ]
                if not func_names:
                    skipped += 1
                    continue
                func_name = func_names[0]
            except SyntaxError:
                skipped += 1
                continue

            # Build test code from assertions
            all_tests = test_list + challenge_test_list
            test_lines = []
            for i, assertion in enumerate(all_tests):
                assertion = assertion.strip()
                if not assertion.startswith("assert"):
                    assertion = f"assert {assertion}"
                test_lines.append(f"def test_{func_name}_{i+1}():\n    {assertion}")

            test_code = "\n\n".join(test_lines)

            # Verify test code parses
            try:
                ast.parse(test_code)
            except SyntaxError:
                skipped += 1
                continue

            examples.append({
                "function_name": func_name,
                "function_code": code,
                "test_code": test_code,
                "features": {},
                "metadata": {"source": "mbpp", "task_id": item["task_id"]},
            })

    print(f"  MBPP: {len(examples)} examples loaded, {skipped} skipped")
    return examples


def load_our_mined(path="datasets/train_filtered.jsonl"):
    """Load our mined dataset, keeping only standalone examples."""
    if not Path(path).exists():
        print(f"  Warning: {path} not found")
        return []

    EXTERNAL_MARKERS = [
        "request.", "self.client", "self.app", "flask.", "django.",
        "db.", "session.", "cursor.", "conn.",
    ]

    with jsonlines.open(path) as reader:
        all_examples = list(reader)

    kept = []
    for ex in all_examples:
        test = ex["test_code"]
        func = ex["function_code"]

        # Skip examples with heavy external deps
        has_ext = any(m in test or m in func for m in EXTERNAL_MARKERS)
        if has_ext:
            continue

        # Verify both parse
        try:
            ast.parse(func)
            ast.parse(test)
        except SyntaxError:
            continue

        # Must have assertions
        if "assert" not in test:
            continue

        # Tag source
        ex.setdefault("metadata", {})["source"] = "mined"
        kept.append(ex)

    print(f"  Mined: {len(kept)} standalone examples kept from {len(all_examples)} total")
    return kept


def deduplicate(examples):
    """Remove exact duplicate function codes."""
    seen = set()
    unique = []
    for ex in examples:
        key = ex["function_code"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def main():
    print("=" * 60)
    print("Building combined training dataset")
    print("=" * 60)

    all_examples = []

    # 1. MBPP
    mbpp = load_mbpp()
    all_examples.extend(mbpp)

    # 2. Our mined data
    mined = load_our_mined()
    all_examples.extend(mined)

    # Deduplicate
    before = len(all_examples)
    all_examples = deduplicate(all_examples)
    print(f"\nDeduplicated: {before} → {len(all_examples)}")

    # Stats
    sources = {}
    for ex in all_examples:
        src = ex.get("metadata", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\nFinal dataset: {len(all_examples)} examples")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Avg lengths
    func_lens = [len(ex["function_code"].split("\n")) for ex in all_examples]
    test_lens = [len(ex["test_code"].split("\n")) for ex in all_examples]
    print(f"\nAvg function length: {sum(func_lens)/len(func_lens):.1f} lines")
    print(f"Avg test length: {sum(test_lens)/len(test_lens):.1f} lines")

    # Save
    out_path = "datasets/train_combined.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_path, mode="w") as writer:
        for ex in all_examples:
            writer.write(ex)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
