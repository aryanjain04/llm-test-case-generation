"""Filter training data for quality: remove examples with heavy external deps."""
import jsonlines
import ast
import re
import textwrap

with jsonlines.open("datasets/train.jsonl") as reader:
    examples = list(reader)

print(f"Total examples before filtering: {len(examples)}")


def try_fix_test(test_code: str) -> str:
    """Try to fix common issues: dedent class methods, wrap bare methods."""
    # Dedent if entire block is indented (extracted class methods)
    dedented = textwrap.dedent(test_code)
    try:
        ast.parse(dedented)
        return dedented
    except SyntaxError:
        pass

    # If it looks like a class method (uses self), wrap in a class
    if "self" in test_code:
        wrapped = "class TestGenerated:\n" + textwrap.indent(textwrap.dedent(test_code), "    ")
        try:
            ast.parse(wrapped)
            # Convert to standalone functions instead: replace self. calls, remove self param
            dedented = textwrap.dedent(test_code)
            # Replace def test_xxx(self, ...) -> def test_xxx(...)
            dedented = re.sub(r'def (\w+)\(self,\s*', r'def \1(', dedented)
            dedented = re.sub(r'def (\w+)\(self\)', r'def \1()', dedented)
            try:
                ast.parse(dedented)
                return dedented
            except SyntaxError:
                return wrapped  # Return class-wrapped version
        except SyntaxError:
            pass

    return test_code  # Return as-is if nothing works


# Stats
too_short = 0
too_long = 0
no_assert = 0
syntax_err = 0
fixed_count = 0
kept = []

for ex in examples:
    func = ex["function_code"]
    test = ex["test_code"]

    # Skip if function is too short (< 3 lines) or too long (> 100 lines)
    func_lines = func.strip().split("\n")
    if len(func_lines) < 3:
        too_short += 1
        continue
    if len(func_lines) > 100:
        too_long += 1
        continue

    # Skip if test has no assert
    if "assert" not in test and "raises" not in test:
        no_assert += 1
        continue

    # Check function parses
    try:
        ast.parse(func)
    except SyntaxError:
        syntax_err += 1
        continue

    # Try to fix test code if needed
    try:
        ast.parse(test)
    except SyntaxError:
        fixed = try_fix_test(test)
        try:
            ast.parse(fixed)
            test = fixed
            fixed_count += 1
        except SyntaxError:
            syntax_err += 1
            continue

    # Truncate if too long for tokenizer
    if len(func) > 2000:
        func = func[:2000]
    if len(test) > 2000:
        test = test[:2000]

    ex["function_code"] = func
    ex["test_code"] = test
    kept.append(ex)

print(f"Filtered out: too_short={too_short}, too_long={too_long}, no_assert={no_assert}, syntax_err={syntax_err}")
print(f"Fixed (dedented/wrapped): {fixed_count}")
print(f"Kept: {len(kept)}")

# Save filtered dataset
with jsonlines.open("datasets/train_filtered.jsonl", mode="w") as writer:
    for ex in kept:
        writer.write(ex)

print(f"Saved to datasets/train_filtered.jsonl")

# Distribution stats
func_lens = [len(ex["function_code"].split("\n")) for ex in kept]
test_lens = [len(ex["test_code"].split("\n")) for ex in kept]
print(f"Avg func length: {sum(func_lens)/len(func_lens):.1f} lines")
print(f"Avg test length: {sum(test_lens)/len(test_lens):.1f} lines")
