"""
AST-based feature extraction for Python source code.

Extracts structural features from Python functions that serve as:
1. Input features for the KAN critic network (value function)
2. Code characterization for dataset analysis
3. Complexity metrics for evaluation stratification

Features extracted:
- Cyclomatic complexity (via radon)
- LOC (lines of code)
- Parameter count
- Return statement count
- Branch count (if/elif/else)
- Loop count (for/while)
- Nesting depth (max)
- Exception handling presence
- Recursion detection
- Number of function calls
- Number of assertions (for test code)
"""

import ast
import textwrap
from dataclasses import dataclass, asdict
from typing import Optional

from radon.complexity import cc_visit
from radon.metrics import mi_visit


@dataclass
class CodeFeatures:
    """Feature vector for a single Python function."""
    name: str
    loc: int                    # lines of code
    param_count: int            # number of parameters
    return_count: int           # number of return statements
    branch_count: int           # if/elif/else branches
    loop_count: int             # for + while loops
    max_nesting_depth: int      # deepest nesting level
    has_exception_handling: bool # try/except present
    is_recursive: bool          # calls itself
    function_call_count: int    # total function/method calls
    has_default_params: bool    # any parameter has a default value
    cyclomatic_complexity: int  # McCabe complexity
    maintainability_index: float # Radon MI score
    assertion_count: int        # number of assert statements (useful for test code)

    def to_vector(self) -> list[float]:
        """Convert to a numeric vector for model input."""
        return [
            float(self.loc),
            float(self.param_count),
            float(self.return_count),
            float(self.branch_count),
            float(self.loop_count),
            float(self.max_nesting_depth),
            float(self.has_exception_handling),
            float(self.is_recursive),
            float(self.function_call_count),
            float(self.has_default_params),
            float(self.cyclomatic_complexity),
            float(self.maintainability_index),
            float(self.assertion_count),
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "loc", "param_count", "return_count", "branch_count",
            "loop_count", "max_nesting_depth", "has_exception_handling",
            "is_recursive", "function_call_count", "has_default_params",
            "cyclomatic_complexity", "maintainability_index", "assertion_count",
        ]

    @staticmethod
    def num_features() -> int:
        return 13


class _NestingVisitor(ast.NodeVisitor):
    """Walks AST to compute max nesting depth."""

    NESTING_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

    def __init__(self):
        self.max_depth = 0
        self._current_depth = 0

    def _visit_nesting(self, node):
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_If(self, node):
        self._visit_nesting(node)

    def visit_For(self, node):
        self._visit_nesting(node)

    def visit_While(self, node):
        self._visit_nesting(node)

    def visit_With(self, node):
        self._visit_nesting(node)

    def visit_Try(self, node):
        self._visit_nesting(node)


class _FeatureVisitor(ast.NodeVisitor):
    """Extracts all features from a function AST node."""

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.return_count = 0
        self.branch_count = 0
        self.loop_count = 0
        self.has_exception_handling = False
        self.is_recursive = False
        self.function_call_count = 0
        self.assertion_count = 0

    def visit_Return(self, node):
        self.return_count += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.branch_count += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.has_exception_handling = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.has_exception_handling = True
        self.generic_visit(node)

    def visit_Call(self, node):
        self.function_call_count += 1
        # Check for recursion
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            self.is_recursive = True
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.assertion_count += 1
        self.generic_visit(node)


def _get_func_source_lines(source: str, node: ast.FunctionDef) -> int:
    """Count lines of code in a function (excluding blank lines and comments)."""
    lines = source.split("\n")
    func_lines = lines[node.lineno - 1 : node.end_lineno]
    loc = 0
    for line in func_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            loc += 1
    return loc


def extract_features(func_node: ast.FunctionDef, source: str) -> CodeFeatures:
    """
    Extract features from an ast.FunctionDef node.

    Args:
        func_node: The AST node for the function
        source: The complete source code (needed for LOC and radon)

    Returns:
        CodeFeatures dataclass with all extracted features
    """
    func_name = func_node.name

    # LOC
    loc = _get_func_source_lines(source, func_node)

    # Parameters
    args = func_node.args
    param_count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
    if args.vararg:
        param_count += 1
    if args.kwarg:
        param_count += 1

    has_default_params = len(args.defaults) > 0 or len(args.kw_defaults) > 0

    # Walk AST for feature counts
    feature_visitor = _FeatureVisitor(func_name)
    feature_visitor.visit(func_node)

    # Nesting depth
    nesting_visitor = _NestingVisitor()
    nesting_visitor.visit(func_node)

    # Radon metrics (need the function source as a string)
    func_source_lines = source.split("\n")[func_node.lineno - 1 : func_node.end_lineno]
    func_source = "\n".join(func_source_lines)

    try:
        cc_results = cc_visit(func_source)
        cyclomatic = cc_results[0].complexity if cc_results else 1
    except Exception:
        cyclomatic = 1

    try:
        mi = mi_visit(func_source, multi=False)
    except Exception:
        mi = 50.0  # default middle-of-the-road value

    return CodeFeatures(
        name=func_name,
        loc=loc,
        param_count=param_count,
        return_count=feature_visitor.return_count,
        branch_count=feature_visitor.branch_count,
        loop_count=feature_visitor.loop_count,
        max_nesting_depth=nesting_visitor.max_depth,
        has_exception_handling=feature_visitor.has_exception_handling,
        is_recursive=feature_visitor.is_recursive,
        function_call_count=feature_visitor.function_call_count,
        has_default_params=has_default_params,
        cyclomatic_complexity=cyclomatic,
        maintainability_index=mi,
        assertion_count=feature_visitor.assertion_count,
    )


def extract_features_from_source(source: str) -> list[CodeFeatures]:
    """
    Extract features for ALL functions in a source code string.

    Args:
        source: Python source code as a string

    Returns:
        List of CodeFeatures, one per function found
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    features = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                feat = extract_features(node, source)
                features.append(feat)
            except Exception as e:
                print(f"Warning: Could not extract features for {node.name}: {e}")
    return features


def extract_functions_from_source(source: str) -> list[tuple[str, str]]:
    """
    Extract individual function source code from a module.

    Args:
        source: Complete Python source code

    Returns:
        List of (function_name, function_source_code) tuples
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    lines = source.split("\n")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_lines = lines[node.lineno - 1 : node.end_lineno]
            func_source = "\n".join(func_lines)
            functions.append((node.name, func_source))
    return functions


# --- CLI for quick testing ---
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.ast_analysis.feature_extractor <file.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    features = extract_features_from_source(source)
    for feat in features:
        print(json.dumps(feat.to_dict(), indent=2))
    print(f"\n--- {len(features)} functions analyzed ---")
    print(f"Feature vector size: {CodeFeatures.num_features()}")
