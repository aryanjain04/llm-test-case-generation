"""
Build the final high-quality training dataset by combining:
1. BigCodeBench — 1,140 examples with proper unittest test classes
2. MBPP — 962 examples, reformatted into proper def test_ functions
3. Our mined standalone examples — 312 clean examples

All outputs are standardized to proper pytest-style test functions.
"""
import json
import re
import ast
import textwrap
from pathlib import Path
from collections import Counter

try:
    import jsonlines
except ImportError:
    raise ImportError("pip install jsonlines")

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("pip install datasets")


# ========================================
# 1. BigCodeBench
# ========================================
def load_bigcodebench():
    """
    Load BigCodeBench and extract function→test pairs.
    
    BigCodeBench has:
    - code_prompt: imports + function signature
    - canonical_solution: function body
    - test: unittest.TestCase class with test methods
    - entry_point: function name
    - libs: required libraries
    
    We combine code_prompt + canonical_solution to get the full function,
    and keep test as-is (it's already proper test code).
    
    We SKIP examples that require heavy external libs (matplotlib, sklearn, etc.)
    since those won't help the model learn basic test generation.
    """
    print("Loading BigCodeBench from HuggingFace...")
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")

    # Libraries that are fine (stdlib + common pure-python)
    ALLOWED_LIBS = {
        "random", "itertools", "collections", "functools", "math", "string",
        "re", "os", "sys", "json", "csv", "datetime", "time", "hashlib",
        "base64", "copy", "operator", "statistics", "decimal", "fractions",
        "textwrap", "io", "pathlib", "typing", "abc", "dataclasses",
        "unittest", "pytest", "struct", "bisect", "heapq", "queue",
        "threading", "logging", "argparse", "configparser", "glob",
        "shutil", "tempfile", "zipfile", "gzip", "pickle", "shelve",
        "sqlite3", "html", "xml", "email", "urllib", "http",
        "numpy", "pandas",  # common enough, and the test patterns are still useful
    }

    examples = []
    skipped_libs = 0
    skipped_parse = 0

    for item in ds:
        # Parse libs field (it's a string repr of a list, e.g. "['random', 'itertools']")
        libs_raw = item.get("libs", "[]")
        try:
            libs = ast.literal_eval(libs_raw) if isinstance(libs_raw, str) else libs_raw
        except (ValueError, SyntaxError):
            libs = []

        if libs and not all(lib.split(".")[0] in ALLOWED_LIBS for lib in libs):
            skipped_libs += 1
            continue

        # Build full function code
        # code_prompt has imports + def line, canonical_solution is the body (unindented)
        code_prompt = item["code_prompt"].strip()
        solution = item["canonical_solution"].rstrip()
        # Indent the solution body if it's not already indented
        if solution and not solution.startswith("    "):
            solution = textwrap.indent(solution, "    ")
        full_function = code_prompt + "\n" + solution

        # Get test code (already proper unittest format)
        test_code = item["test"].strip()
        func_name = item["entry_point"]

        # Verify both parse
        try:
            ast.parse(full_function)
        except SyntaxError:
            skipped_parse += 1
            continue

        try:
            ast.parse(test_code)
        except SyntaxError:
            skipped_parse += 1
            continue

        # Strip import lines from function code that would conflict
        # (the function code has imports at the top, which is fine for training)
        
        examples.append({
            "function_name": func_name,
            "function_code": full_function,
            "test_code": test_code,
            "features": {},
            "metadata": {
                "source": "bigcodebench",
                "task_id": item["task_id"],
                "libs": libs,
            },
        })

    print(f"  BigCodeBench: {len(examples)} kept, {skipped_libs} skipped (heavy libs), {skipped_parse} parse errors")
    return examples


# ========================================
# 2. MBPP (reformatted)
# ========================================
def load_mbpp_reformatted():
    """
    Load MBPP and reformat assert one-liners into proper def test_ functions.
    
    MBPP original:  assert is_prime(5) == True
    Our format:     def test_is_prime_1():\n    assert is_prime(5) == True
    """
    print("Loading MBPP from HuggingFace...")
    ds = load_dataset("mbpp", "full")

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

            # Verify function code parses & extract function name
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

            # Build proper pytest test functions from assert statements
            all_tests = test_list + challenge_test_list
            test_lines = []
            for i, assertion in enumerate(all_tests):
                assertion = assertion.strip()
                if not assertion.startswith("assert"):
                    assertion = f"assert {assertion}"
                # Create a proper test function
                test_lines.append(
                    f"def test_{func_name}_{i+1}():\n    {assertion}"
                )

            test_code = "\n\n\n".join(test_lines)

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


# ========================================
# 3. Our mined data (cleaned)
# ========================================
def load_mined(path="datasets/train_filtered.jsonl"):
    """Load our mined dataset, keeping only standalone examples."""
    if not Path(path).exists():
        print(f"  Warning: {path} not found, skipping mined data")
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

        has_ext = any(m in test or m in func for m in EXTERNAL_MARKERS)
        if has_ext:
            continue

        try:
            ast.parse(func)
            ast.parse(test)
        except SyntaxError:
            continue

        if "assert" not in test:
            continue

        ex.setdefault("metadata", {})["source"] = "mined"
        kept.append(ex)

    print(f"  Mined: {len(kept)} standalone examples from {len(all_examples)} total")
    return kept


# ========================================
# Combine & Save
# ========================================
def deduplicate(examples):
    """Remove exact-duplicate function codes."""
    seen = set()
    unique = []
    for ex in examples:
        key = ex["function_code"].strip()[:500]  # first 500 chars as key
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def truncate_for_tokenizer(examples, max_chars=2000):
    """Truncate overly long examples to fit in 512 tokens (~4 chars/token)."""
    truncated = 0
    for ex in examples:
        if len(ex["function_code"]) > max_chars:
            ex["function_code"] = ex["function_code"][:max_chars]
            truncated += 1
        if len(ex["test_code"]) > max_chars:
            ex["test_code"] = ex["test_code"][:max_chars]
            truncated += 1
    if truncated:
        print(f"  Truncated {truncated} fields to {max_chars} chars")
    return examples


def main():
    print("=" * 60)
    print("Building Final Combined Training Dataset")
    print("=" * 60)
    print()

    all_examples = []

    # 1. BigCodeBench
    bcb = load_bigcodebench()
    all_examples.extend(bcb)

    # 2. MBPP (reformatted)
    mbpp = load_mbpp_reformatted()
    all_examples.extend(mbpp)

    # 3. Our mined data
    mined = load_mined()
    all_examples.extend(mined)

    # Deduplicate
    before = len(all_examples)
    all_examples = deduplicate(all_examples)
    print(f"\nDeduplicated: {before} → {len(all_examples)}")

    # Truncate
    all_examples = truncate_for_tokenizer(all_examples)

    # Final stats
    sources = Counter()
    for ex in all_examples:
        src = ex.get("metadata", {}).get("source", "unknown")
        sources[src] += 1

    func_lens = [len(ex["function_code"].split("\n")) for ex in all_examples]
    test_lens = [len(ex["test_code"].split("\n")) for ex in all_examples]

    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(all_examples)} examples")
    print(f"{'='*60}")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"\nAvg function length: {sum(func_lens)/len(func_lens):.1f} lines")
    print(f"Avg test length: {sum(test_lens)/len(test_lens):.1f} lines")
    print(f"Min/Max func lines: {min(func_lens)}/{max(func_lens)}")
    print(f"Min/Max test lines: {min(test_lens)}/{max(test_lens)}")

    # Save
    out_path = "datasets/train_combined.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_path, mode="w") as writer:
        for ex in all_examples:
            writer.write(ex)

    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")
    print("Ready to upload to Kaggle!")


if __name__ == "__main__":
    main()
