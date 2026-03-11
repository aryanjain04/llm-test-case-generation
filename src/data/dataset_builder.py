"""
Dataset builder for function → test case pairs.

Two data sources:
1. GitHub mining: Clone repos, extract function-test pairs via AST matching
2. Methods2Test-style: Load pre-built datasets

Output format: JSONL with fields:
  - function_code: str
  - test_code: str
  - function_name: str
  - features: dict (AST features)
  - metadata: dict (repo, file, etc.)
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import jsonlines
from tqdm import tqdm

from src.ast_analysis.feature_extractor import (
    extract_features_from_source,
    extract_functions_from_source,
)


@dataclass
class FunctionTestPair:
    """A single function → test mapping."""
    function_name: str
    function_code: str
    test_code: str
    features: dict
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetBuilder:
    """
    Build a dataset of function-test pairs from Python repositories.

    Strategy:
    1. Walk repo looking for test files (test_*.py, *_test.py)
    2. For each test file, find the module it tests
    3. Match test functions to source functions by name heuristics
    4. Extract AST features for each function
    """

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs: list[FunctionTestPair] = []

    def process_repo(self, repo_path: str) -> list[FunctionTestPair]:
        """
        Process a single repository to extract function-test pairs.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            List of extracted FunctionTestPair objects
        """
        repo_path = Path(repo_path)
        pairs = []

        # Find all test files
        test_files = list(repo_path.rglob("test_*.py")) + list(repo_path.rglob("*_test.py"))

        for test_file in test_files:
            try:
                test_source = test_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Find the source module this test file corresponds to
            source_file = self._find_source_module(test_file, repo_path)
            if source_file is None:
                continue

            try:
                source_code = source_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Extract function-test mappings
            file_pairs = self._match_functions_to_tests(
                source_code, test_source,
                source_file=str(source_file.relative_to(repo_path)),
                test_file=str(test_file.relative_to(repo_path)),
                repo_name=repo_path.name,
            )
            pairs.extend(file_pairs)

        self.pairs.extend(pairs)
        return pairs

    def _find_source_module(self, test_file: Path, repo_root: Path) -> Optional[Path]:
        """
        Find the source module that a test file corresponds to.

        Strategy:
        1. Try direct name-based candidates (test_foo.py → foo.py)
        2. Parse test file imports to find the source module
        3. Recursive glob as last resort
        """
        test_name = test_file.stem  # e.g., "test_utils"

        # Remove test_ prefix or _test suffix
        if test_name.startswith("test_"):
            module_name = test_name[5:]  # "utils"
        elif test_name.endswith("_test"):
            module_name = test_name[:-5]
        else:
            return None

        # Strategy 1: Direct path candidates (ordered by likelihood)
        candidates = [
            test_file.parent / f"{module_name}.py",
            test_file.parent.parent / f"{module_name}.py",
            test_file.parent.parent / "src" / f"{module_name}.py",
            repo_root / "src" / f"{module_name}.py",
            repo_root / f"{module_name}.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Strategy 2: Recursive glob - find foo.py anywhere in the repo
        matches = list(repo_root.rglob(f"{module_name}.py"))
        # Filter out test files, __pycache__, .venv, etc.
        matches = [
            m for m in matches
            if "test" not in m.name.lower()
            and "__pycache__" not in str(m)
            and ".venv" not in str(m)
            and "venv" not in str(m)
        ]
        if matches:
            return matches[0]

        # Strategy 3: Parse imports from the test file to find source module
        try:
            test_source = test_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(test_source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # e.g., "from mypackage.utils import foo" → look for utils.py
                    parts = node.module.split(".")
                    for part in parts:
                        part_matches = list(repo_root.rglob(f"{part}.py"))
                        part_matches = [
                            m for m in part_matches
                            if "test" not in m.name.lower()
                            and "__pycache__" not in str(m)
                        ]
                        if part_matches:
                            return part_matches[0]
        except Exception:
            pass

        return None

    def _match_functions_to_tests(
        self,
        source_code: str,
        test_code: str,
        source_file: str,
        test_file: str,
        repo_name: str,
    ) -> list[FunctionTestPair]:
        """
        Match source functions to their test functions by name heuristics.

        Matching rules:
        - test_add() → add()
        - TestAdd → add()
        - test_add_positive_numbers() → add()
        """
        # Extract source functions
        source_functions = extract_functions_from_source(source_code)
        if not source_functions:
            return []

        # Extract test functions/classes
        try:
            test_tree = ast.parse(test_code)
        except SyntaxError:
            return []

        test_lines = test_code.split("\n")
        source_features = extract_features_from_source(source_code)
        feature_map = {f.name: f.to_dict() for f in source_features}

        pairs = []
        func_name_set = {name for name, _ in source_functions}

        for func_name, func_source in source_functions:
            # Find all test functions that match this source function
            matching_tests = self._find_matching_tests(
                func_name, test_tree, test_lines
            )

            if matching_tests:
                combined_test = "\n\n".join(matching_tests)
                features = feature_map.get(func_name, {})

                pair = FunctionTestPair(
                    function_name=func_name,
                    function_code=func_source,
                    test_code=combined_test,
                    features=features,
                    metadata={
                        "repo": repo_name,
                        "source_file": source_file,
                        "test_file": test_file,
                        "num_test_functions": len(matching_tests),
                    },
                )
                pairs.append(pair)

        return pairs

    def _find_matching_tests(
        self, func_name: str, test_tree: ast.Module, test_lines: list[str]
    ) -> list[str]:
        """
        Find test functions that test a given source function.

        Matching strategies:
        1. Name-based: test_funcname() → funcname()
        2. Call-based: any test function that calls funcname()
        """
        matching = []
        func_name_lower = func_name.lower()

        for node in ast.walk(test_tree):
            if isinstance(node, ast.FunctionDef):
                test_name_lower = node.name.lower()

                # Strategy 1: Name-based matching
                name_match = (
                    test_name_lower == f"test_{func_name_lower}"
                    or test_name_lower.startswith(f"test_{func_name_lower}_")
                )

                # Strategy 2: Call-based matching (test calls the source function)
                call_match = False
                if not name_match and node.name.startswith("test"):
                    call_match = self._test_calls_function(node, func_name)

                if name_match or call_match:
                    func_lines = test_lines[node.lineno - 1 : node.end_lineno]
                    matching.append("\n".join(func_lines))

            elif isinstance(node, ast.ClassDef):
                class_name_lower = node.name.lower()
                # Match TestFuncName class
                if class_name_lower == f"test{func_name_lower}":
                    func_lines = test_lines[node.lineno - 1 : node.end_lineno]
                    matching.append("\n".join(func_lines))

        return matching

    def _test_calls_function(self, test_node: ast.FunctionDef, func_name: str) -> bool:
        """Check if a test function body contains a call to the given function."""
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                # Direct call: func_name(...)
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True
                # Attribute call: module.func_name(...)
                if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                    return True
        return False

    def save(self, filename: str = "train.jsonl"):
        """Save collected pairs to JSONL file."""
        output_path = self.output_dir / filename
        with jsonlines.open(output_path, mode="w") as writer:
            for pair in self.pairs:
                writer.write(pair.to_dict())
        print(f"Saved {len(self.pairs)} pairs to {output_path}")

    def load(self, filename: str = "train.jsonl") -> list[FunctionTestPair]:
        """Load pairs from JSONL file."""
        input_path = self.output_dir / filename
        self.pairs = []
        with jsonlines.open(input_path) as reader:
            for obj in reader:
                self.pairs.append(FunctionTestPair(**obj))
        return self.pairs

    def stats(self) -> dict:
        """Print dataset statistics."""
        if not self.pairs:
            return {"total_pairs": 0}

        avg_func_loc = sum(
            p.features.get("loc", 0) for p in self.pairs
        ) / len(self.pairs)
        avg_test_lines = sum(
            len(p.test_code.split("\n")) for p in self.pairs
        ) / len(self.pairs)

        stats = {
            "total_pairs": len(self.pairs),
            "avg_function_loc": round(avg_func_loc, 1),
            "avg_test_lines": round(avg_test_lines, 1),
            "unique_repos": len(set(p.metadata.get("repo", "") for p in self.pairs)),
        }
        print(json.dumps(stats, indent=2))
        return stats


class GitHubMiner:
    """
    Mine Python repos from GitHub for function-test pairs.

    Usage:
        miner = GitHubMiner(clone_dir="repos_cache")
        miner.clone_repos(["pallets/flask", "psf/requests", ...])
        builder = DatasetBuilder()
        for repo_path in miner.get_repos():
            builder.process_repo(repo_path)
        builder.save("train.jsonl")
    """

    # Curated list of well-tested Python repos
    RECOMMENDED_REPOS = [
        "pallets/flask",
        "psf/requests",
        "python-attrs/attrs",
        "more-itertools/more-itertools",
        "dateutil/dateutil",
        "simplejson/simplejson",
        "benjaminp/six",
        "pypa/packaging",
        "jaraco/path",
        "agronholm/anyio",
        "pallets/click",
        "pallets/jinja",
        "pallets/markupsafe",
        "pallets/itsdangerous",
        "pytest-dev/pluggy",
        "hynek/structlog",
        "mahmoud/boltons",
        "aio-libs/aiohttp",
        "encode/httpx",
        "encode/starlette",
    ]

    def __init__(self, clone_dir: str = "repos_cache"):
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)

    def clone_repos(self, repo_list: Optional[list[str]] = None, max_repos: int = 20):
        """Clone repos from GitHub. Requires git to be installed."""
        import git

        repos = repo_list or self.RECOMMENDED_REPOS[:max_repos]

        for repo_url in tqdm(repos, desc="Cloning repos"):
            repo_name = repo_url.split("/")[-1]
            dest = self.clone_dir / repo_name

            if dest.exists():
                print(f"  {repo_name}: already exists, skipping")
                continue

            try:
                full_url = f"https://github.com/{repo_url}.git"
                print(f"  Cloning {repo_url}...")
                git.Repo.clone_from(
                    full_url,
                    str(dest),
                    depth=1,  # shallow clone to save space
                    single_branch=True,
                )
            except Exception as e:
                print(f"  Failed to clone {repo_url}: {e}")

    def get_repos(self) -> list[Path]:
        """Return list of cloned repo directories."""
        return [d for d in self.clone_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]


# --- CLI ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mine":
        print("=== Mining GitHub repos ===")
        miner = GitHubMiner()
        max_repos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        miner.clone_repos(max_repos=max_repos)

        builder = DatasetBuilder()
        for repo_path in miner.get_repos():
            print(f"\nProcessing {repo_path.name}...")
            pairs = builder.process_repo(str(repo_path))
            print(f"  Found {len(pairs)} pairs")

        builder.save("train.jsonl")
        builder.stats()
    else:
        print("Usage:")
        print("  python -m src.data.dataset_builder mine [max_repos]")
        print("  Example: python -m src.data.dataset_builder mine 5")
