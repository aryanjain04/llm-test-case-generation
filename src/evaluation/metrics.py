"""
Evaluation metrics for generated test cases.

Metrics:
1. Compilation Rate: % of generated tests that parse without SyntaxError
2. Execution Pass Rate: % of generated tests that pass all assertions
3. Line Coverage: achieved by generated tests on source functions
4. Branch Coverage: achieved by generated tests on source functions
5. Mutation Score: % of injected mutants killed by generated tests
6. BLEU / CodeBLEU: similarity to reference test cases (when available)
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.execution.sandbox import TestExecutor, ExecutionResult


@dataclass
class EvaluationResult:
    """Aggregated evaluation metrics across a set of functions."""
    total_functions: int = 0
    compilable_count: int = 0
    compilable_rate: float = 0.0
    avg_pass_rate: float = 0.0
    avg_line_coverage: float = 0.0
    avg_branch_coverage: float = 0.0
    avg_bleu: float = 0.0
    avg_reward: float = 0.0
    per_function: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"=== Evaluation Summary ===\n"
            f"Functions evaluated: {self.total_functions}\n"
            f"Compilable: {self.compilable_count}/{self.total_functions} "
            f"({self.compilable_rate:.1f}%)\n"
            f"Avg Pass Rate: {self.avg_pass_rate:.1f}%\n"
            f"Avg Line Coverage: {self.avg_line_coverage:.1f}%\n"
            f"Avg Branch Coverage: {self.avg_branch_coverage:.1f}%\n"
            f"Avg BLEU: {self.avg_bleu:.4f}\n"
            f"Avg Reward: {self.avg_reward:.4f}\n"
        )


class Evaluator:
    """
    End-to-end evaluator for test generation quality.

    Usage:
        evaluator = Evaluator()
        result = evaluator.evaluate(
            functions=[(name, source), ...],
            generated_tests=[(name, test_code), ...],
            reference_tests=[(name, ref_code), ...],  # optional
        )
        print(result.summary())
    """

    def __init__(self, timeout: int = 30):
        self.executor = TestExecutor(timeout=timeout)
        # Download NLTK data for BLEU
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def evaluate(
        self,
        functions: list[tuple[str, str]],
        generated_tests: list[tuple[str, str]],
        reference_tests: Optional[list[tuple[str, str]]] = None,
    ) -> EvaluationResult:
        """
        Evaluate generated tests against source functions.

        Args:
            functions: List of (function_name, function_source_code)
            generated_tests: List of (function_name, generated_test_code)
            reference_tests: Optional list of (function_name, reference_test_code) for BLEU

        Returns:
            EvaluationResult with aggregated metrics
        """
        func_map = {name: code for name, code in functions}
        gen_map = {name: code for name, code in generated_tests}
        ref_map = {}
        if reference_tests:
            ref_map = {name: code for name, code in reference_tests}

        result = EvaluationResult()
        result.total_functions = len(functions)

        total_pass_rate = 0.0
        total_line_cov = 0.0
        total_branch_cov = 0.0
        total_bleu = 0.0
        total_reward = 0.0
        compilable = 0
        bleu_count = 0

        for func_name, func_source in functions:
            test_source = gen_map.get(func_name, "")
            if not test_source:
                continue

            # Execute the generated test
            exec_result = self.executor.execute(func_source, test_source)

            if exec_result.compilable:
                compilable += 1

            total_pass_rate += exec_result.pass_rate * 100
            total_line_cov += exec_result.line_coverage
            total_branch_cov += exec_result.branch_coverage
            total_reward += exec_result.reward

            # BLEU score against reference
            bleu = 0.0
            if func_name in ref_map:
                bleu = self._compute_bleu(test_source, ref_map[func_name])
                total_bleu += bleu
                bleu_count += 1

            per_func = {
                "function_name": func_name,
                "compilable": exec_result.compilable,
                "passed": exec_result.passed,
                "failed": exec_result.failed,
                "total_tests": exec_result.total,
                "pass_rate": exec_result.pass_rate,
                "line_coverage": exec_result.line_coverage,
                "branch_coverage": exec_result.branch_coverage,
                "bleu": bleu,
                "reward": exec_result.reward,
                "errors": exec_result.error_messages[:3],
            }
            result.per_function.append(per_func)

        n = max(result.total_functions, 1)
        result.compilable_count = compilable
        result.compilable_rate = (compilable / n) * 100
        result.avg_pass_rate = total_pass_rate / n
        result.avg_line_coverage = total_line_cov / n
        result.avg_branch_coverage = total_branch_cov / n
        result.avg_bleu = total_bleu / max(bleu_count, 1)
        result.avg_reward = total_reward / n

        return result

    def _compute_bleu(self, generated: str, reference: str) -> float:
        """Compute BLEU score between generated and reference test code."""
        gen_tokens = generated.split()
        ref_tokens = reference.split()

        if not gen_tokens or not ref_tokens:
            return 0.0

        smoothing = SmoothingFunction().method1
        try:
            return sentence_bleu(
                [ref_tokens],
                gen_tokens,
                smoothing_function=smoothing,
            )
        except Exception:
            return 0.0

    def save_results(self, result: EvaluationResult, filepath: str = "results/eval.json"):
        """Save evaluation results to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {path}")


# --- CLI ---
if __name__ == "__main__":
    print("Evaluator module loaded. Use Evaluator class in scripts.")
    print("See scripts/run_baseline.py for usage example.")
