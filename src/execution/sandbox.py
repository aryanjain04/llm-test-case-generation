"""
Sandboxed test execution engine.

Executes generated test code safely, captures:
1. Pass/fail status
2. Line & branch coverage
3. Error messages
4. Execution time

This provides the REWARD SIGNAL for RL training:
- reward = α * coverage + β * pass_rate + γ * mutation_score
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing a generated test suite."""
    passed: int           # number of tests passed
    failed: int           # number of tests failed
    errors: int           # number of tests errored (syntax, import, etc.)
    total: int            # total test count
    pass_rate: float      # passed / total (0.0 - 1.0)
    line_coverage: float  # % of source lines covered (0.0 - 100.0)
    branch_coverage: float # % of branches covered (0.0 - 100.0)
    execution_time: float  # seconds
    error_messages: list[str]  # stderr/assertion errors
    compilable: bool      # did the test code at least parse?

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reward(self) -> float:
        """
        Compute RL reward signal.

        Reward = 0.0 if not compilable
        Otherwise: weighted combination of coverage and pass rate
        """
        if not self.compilable:
            return -1.0

        if self.total == 0:
            return -0.5  # generated no actual tests

        # Weights (tune these during RL phase)
        alpha = 0.4  # coverage weight
        beta = 0.4   # pass rate weight
        gamma = 0.2  # branch coverage bonus

        coverage_norm = self.line_coverage / 100.0
        branch_norm = self.branch_coverage / 100.0

        return alpha * coverage_norm + beta * self.pass_rate + gamma * branch_norm


class TestExecutor:
    """
    Execute generated tests in an isolated environment.

    Uses subprocess + tempfile to prevent interference
    between test runs and protect the host system.
    """

    def __init__(self, timeout: int = 30, python_executable: Optional[str] = None):
        self.timeout = timeout
        if python_executable:
            self.python_executable = python_executable
        else:
            # Prefer venv python if we're in one
            self.python_executable = self._find_python()

    @staticmethod
    def _find_python() -> str:
        """Find the correct Python executable, preferring venv."""
        # Check if we're in a venv
        venv_dir = os.environ.get("VIRTUAL_ENV")
        if venv_dir:
            if sys.platform == "win32":
                candidate = os.path.join(venv_dir, "Scripts", "python.exe")
            else:
                candidate = os.path.join(venv_dir, "bin", "python")
            if os.path.exists(candidate):
                return candidate

        # Check project-local .venv
        project_root = Path(__file__).parent.parent.parent
        local_venv = project_root / ".venv"
        if local_venv.exists():
            if sys.platform == "win32":
                candidate = str(local_venv / "Scripts" / "python.exe")
            else:
                candidate = str(local_venv / "bin" / "python")
            if os.path.exists(candidate):
                return candidate

        return sys.executable

    def execute(
        self,
        function_source: str,
        test_source: str,
        function_name: str = "functions_under_test",
    ) -> ExecutionResult:
        """
        Execute generated test code against source function code.

        Args:
            function_source: The source code of the function(s) being tested
            test_source: The generated test code (pytest format)
            function_name: Module name for the function file

        Returns:
            ExecutionResult with all metrics
        """
        with tempfile.TemporaryDirectory(prefix="testgen_") as tmpdir:
            tmpdir = Path(tmpdir)

            # Write the function under test
            func_file = tmpdir / f"{function_name}.py"
            func_file.write_text(function_source, encoding="utf-8")

            # Fix imports in test source - replace absolute imports with local
            fixed_test = self._fix_imports(test_source, function_name, function_source)

            # Check if test code is syntactically valid
            compilable = self._check_syntax(fixed_test)
            if not compilable:
                return ExecutionResult(
                    passed=0, failed=0, errors=1, total=0,
                    pass_rate=0.0, line_coverage=0.0, branch_coverage=0.0,
                    execution_time=0.0,
                    error_messages=["Generated test code has syntax errors"],
                    compilable=False,
                )

            # Write test file
            test_file = tmpdir / f"test_{function_name}.py"
            test_file.write_text(fixed_test, encoding="utf-8")

            # Run with coverage
            return self._run_with_coverage(tmpdir, func_file, test_file, function_name)

    def _fix_imports(self, test_source: str, module_name: str, function_source: str = "") -> str:
        """Adjust imports so tests can find the function module in tmpdir.
        
        If no import statement is found in the test source, auto-injects
        a wildcard import from the function module.
        """
        # Replace common import patterns
        import_patterns = [
            (f"from functions import", f"from {module_name} import"),
            (f"import functions", f"import {module_name}"),
        ]
        result = test_source
        for old, new in import_patterns:
            result = result.replace(old, new)

        # Check if there's any import of the module already
        has_import = (
            f"from {module_name} import" in result
            or f"import {module_name}" in result
        )

        if not has_import:
            # Extract function names from source to build explicit imports
            import ast
            try:
                tree = ast.parse(function_source)
                func_names = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                if func_names:
                    import_line = f"from {module_name} import {', '.join(func_names)}\n"
                else:
                    import_line = f"from {module_name} import *\n"
            except SyntaxError:
                import_line = f"from {module_name} import *\n"

            # Prepend import (after any existing imports/comments at top)
            result = import_line + result

        return result

    def _check_syntax(self, code: str) -> bool:
        """Check if Python code is syntactically valid."""
        try:
            compile(code, "<test>", "exec")
            return True
        except SyntaxError:
            return False

    def _run_with_coverage(
        self,
        tmpdir: Path,
        func_file: Path,
        test_file: Path,
        module_name: str,
    ) -> ExecutionResult:
        """Run pytest with coverage and parse results."""
        start_time = time.time()

        # Run pytest with coverage and JSON output
        cmd = [
            self.python_executable, "-m", "pytest",
            str(test_file),
            f"--tb=short",
            "-q",
            f"--cov={module_name}",
            "--cov-branch",
            "--cov-report=json",
            "--no-header",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(tmpdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                passed=0, failed=0, errors=1, total=0,
                pass_rate=0.0, line_coverage=0.0, branch_coverage=0.0,
                execution_time=self.timeout,
                error_messages=["Test execution timed out"],
                compilable=True,
            )

        execution_time = time.time() - start_time

        # Parse pytest output
        passed, failed, errors = self._parse_pytest_output(result.stdout)
        total = passed + failed + errors
        pass_rate = passed / total if total > 0 else 0.0

        # Parse coverage
        line_cov, branch_cov = self._parse_coverage(tmpdir)

        # Collect error messages
        error_msgs = []
        if result.returncode != 0:
            stderr_lines = result.stderr.strip().split("\n") if result.stderr else []
            stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
            error_msgs = [l for l in stderr_lines + stdout_lines if "FAILED" in l or "ERROR" in l or "Error" in l]

        return ExecutionResult(
            passed=passed,
            failed=failed,
            errors=errors,
            total=total,
            pass_rate=pass_rate,
            line_coverage=line_cov,
            branch_coverage=branch_cov,
            execution_time=execution_time,
            error_messages=error_msgs[:10],  # cap at 10
            compilable=True,
        )

    def _parse_pytest_output(self, stdout: str) -> tuple[int, int, int]:
        """Parse pytest short output to get passed/failed/error counts."""
        import re

        passed = failed = errors = 0

        # Look for summary line like "5 passed, 2 failed, 1 error"
        match = re.search(r"(\d+) passed", stdout)
        if match:
            passed = int(match.group(1))

        match = re.search(r"(\d+) failed", stdout)
        if match:
            failed = int(match.group(1))

        match = re.search(r"(\d+) error", stdout)
        if match:
            errors = int(match.group(1))

        return passed, failed, errors

    def _parse_coverage(self, tmpdir: Path) -> tuple[float, float]:
        """Parse coverage.json to extract line and branch coverage."""
        cov_file = tmpdir / "coverage.json"
        if not cov_file.exists():
            return 0.0, 0.0

        try:
            data = json.loads(cov_file.read_text(encoding="utf-8"))
            totals = data.get("totals", {})
            line_cov = totals.get("percent_covered", 0.0)
            branch_cov = totals.get("percent_covered_branches", 0.0)
            return line_cov, branch_cov
        except (json.JSONDecodeError, KeyError):
            return 0.0, 0.0


# --- Quick test ---
if __name__ == "__main__":
    func_code = '''
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''

    test_code = '''
import pytest
from functions_under_test import add, divide

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
'''

    executor = TestExecutor()
    result = executor.execute(func_code, test_code)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nReward: {result.reward:.4f}")
