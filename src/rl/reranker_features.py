"""
Feature extraction for critic-based candidate reranking.

These features are intentionally static (no execution-time metrics) so they
can be computed at inference time for every generated candidate.
"""

from __future__ import annotations

import ast
from typing import Any

from src.ast_analysis.feature_extractor import CodeFeatures, extract_features_from_source


FUNCTION_FEATURE_NAMES = [f"func_{n}" for n in CodeFeatures.feature_names()]
TEST_FEATURE_NAMES = [
    "test_syntax_valid",
    "test_line_count",
    "test_char_count",
    "test_token_count",
    "test_function_count",
    "test_assert_count",
    "test_pytest_raises_count",
    "test_import_count",
    "test_branch_count",
    "test_loop_count",
    "test_fixture_decorator_count",
    "test_parametrize_decorator_count",
    "test_avg_test_func_loc",
    "test_max_test_func_loc",
]
ALL_FEATURE_NAMES = FUNCTION_FEATURE_NAMES + TEST_FEATURE_NAMES


def _count_loc(src: str) -> int:
    return sum(1 for ln in src.splitlines() if ln.strip())


def _safe_attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def extract_function_feature_vector(function_code: str) -> list[float]:
    feats = extract_features_from_source(function_code)
    if not feats:
        return [0.0] * len(FUNCTION_FEATURE_NAMES)
    return feats[0].to_vector()


def extract_test_feature_vector(test_code: str) -> list[float]:
    line_count = float(_count_loc(test_code))
    char_count = float(len(test_code))
    token_count = float(len(test_code.split()))

    try:
        tree = ast.parse(test_code)
        syntax_valid = 1.0
    except SyntaxError:
        # Keep only coarse text-size features when parse fails.
        return [
            0.0,
            line_count,
            char_count,
            token_count,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    test_funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
    ]
    assert_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
    import_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
    branch_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    loop_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))

    pytest_raises_count = 0
    fixture_decorator_count = 0
    parametrize_decorator_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call):
                    fn = _safe_attr_name(ctx.func)
                    if fn == "raises":
                        pytest_raises_count += 1

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                name = _safe_attr_name(dec if not isinstance(dec, ast.Call) else dec.func)
                if name == "fixture":
                    fixture_decorator_count += 1
                if name == "parametrize":
                    parametrize_decorator_count += 1

    func_locs: list[int] = []
    lines = test_code.splitlines()
    for fn in test_funcs:
        if hasattr(fn, "end_lineno") and fn.end_lineno is not None:
            span = lines[fn.lineno - 1 : fn.end_lineno]
            func_locs.append(sum(1 for ln in span if ln.strip()))

    avg_func_loc = float(sum(func_locs) / len(func_locs)) if func_locs else 0.0
    max_func_loc = float(max(func_locs)) if func_locs else 0.0

    return [
        syntax_valid,
        line_count,
        char_count,
        token_count,
        float(len(test_funcs)),
        float(assert_count),
        float(pytest_raises_count),
        float(import_count),
        float(branch_count),
        float(loop_count),
        float(fixture_decorator_count),
        float(parametrize_decorator_count),
        avg_func_loc,
        max_func_loc,
    ]


def build_combined_feature_vector(function_code: str, test_code: str) -> tuple[list[float], list[str]]:
    func_vec = extract_function_feature_vector(function_code)
    test_vec = extract_test_feature_vector(test_code)
    return func_vec + test_vec, ALL_FEATURE_NAMES.copy()


def standardize_features(vectors: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    out: list[list[float]] = []
    for vec in vectors:
        out.append([(x - m) / s for x, m, s in zip(vec, mean, std)])
    return out


def compute_standardization_stats(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        raise ValueError("No vectors provided")

    dims = len(vectors[0])
    n = float(len(vectors))

    mean = [0.0] * dims
    for vec in vectors:
        for i, x in enumerate(vec):
            mean[i] += x
    mean = [m / n for m in mean]

    var = [0.0] * dims
    for vec in vectors:
        for i, x in enumerate(vec):
            d = x - mean[i]
            var[i] += d * d
    var = [v / n for v in var]

    # Avoid divide-by-zero for constant features.
    std = [v ** 0.5 if v > 1e-12 else 1.0 for v in var]
    return mean, std
