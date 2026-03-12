"""
CodeT5-based test case generator.

This module handles:
1. Zero-shot inference (baseline)
2. Loading fine-tuned checkpoints
3. Prompt formatting for function → test generation

Model: Salesforce/codet5-small (60M params, fits 4-6GB VRAM easily)
Upgrade path: codet5-base (220M) or codet5p-220m when resources allow.
"""

import torch

# Fix peft compatibility with PyTorch 2.6 (DTensor moved to torch.distributed._tensor)
if not hasattr(torch.distributed, "tensor"):
    import torch.distributed._tensor
    torch.distributed.tensor = torch.distributed._tensor

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)
from peft import PeftModel
from typing import Optional
from pathlib import Path


# === Prompt Templates ===

PROMPT_TEMPLATE_V1 = """# Generate pytest test cases for the following Python function:

{function_code}

# Test cases using pytest:
"""

PROMPT_TEMPLATE_V2 = """Write comprehensive pytest unit tests for the following Python function.
Include edge cases, boundary conditions, and error handling tests.

```python
{function_code}
```

```python
import pytest
"""

PROMPT_TEMPLATE_MINIMAL = """# Function:
{function_code}

# Unit test:
def test_"""

# This matches the exact prompt used during fine-tuning
PROMPT_TEMPLATE_FINETUNE = """Generate pytest tests for:
{function_code}"""


class CodeT5Generator:
    """Wrapper for CodeT5 test generation with multiple prompt strategies."""

    DEFAULT_MODEL = "Salesforce/codet5-small"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"Loading model: {model_name_or_path}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def generate(
        self,
        function_code: str,
        prompt_template: str = PROMPT_TEMPLATE_V1,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        do_sample: bool = True,
    ) -> list[str]:
        """
        Generate test cases for a given function.

        Args:
            function_code: The Python function source code
            prompt_template: Template with {function_code} placeholder
            num_return_sequences: How many candidates to generate
            temperature: Sampling temperature (higher = more diverse)
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample (True) or greedy decode (False)

        Returns:
            List of generated test code strings
        """
        prompt = prompt_template.format(function_code=function_code.strip())

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated.append(text)

        return generated

    def generate_batch(
        self,
        functions: list[str],
        prompt_template: str = PROMPT_TEMPLATE_V1,
        **kwargs,
    ) -> list[list[str]]:
        """Generate tests for multiple functions."""
        results = []
        for func in functions:
            tests = self.generate(func, prompt_template, **kwargs)
            results.append(tests)
        return results

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        base_model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> "CodeT5Generator":
        """Load from a fine-tuned checkpoint directory (supports LoRA adapters).
        
        Args:
            checkpoint_path: Path to the LoRA adapter checkpoint
            base_model: Base model name to load adapters onto
        """
        instance = cls.__new__(cls)
        instance.device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.max_length = kwargs.get("max_length", 512)

        print(f"Loading base model: {base_model}")
        print(f"Loading LoRA adapter: {checkpoint_path}")
        print(f"Device: {instance.device}")

        instance.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Check if this is a PEFT/LoRA checkpoint
        adapter_config = Path(checkpoint_path) / "adapter_config.json"
        if adapter_config.exists():
            base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            instance.model = PeftModel.from_pretrained(base, checkpoint_path)
            instance.model = instance.model.merge_and_unload()  # Merge for faster inference
            print("LoRA adapter merged into base model")
        else:
            instance.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

        instance.model.to(instance.device)
        instance.model.eval()

        print(f"Model loaded. Parameters: {sum(p.numel() for p in instance.model.parameters()):,}")
        return instance


def run_baseline_demo(functions_file: str):
    """
    Run zero-shot generation on functions from a file.
    This is the very first thing to test — does the model produce anything useful?
    """
    from src.ast_analysis.feature_extractor import extract_functions_from_source

    with open(functions_file, "r", encoding="utf-8") as f:
        source = f.read()

    functions = extract_functions_from_source(source)
    print(f"Found {len(functions)} functions in {functions_file}\n")

    generator = CodeT5Generator()

    all_results = {}
    for func_name, func_source in functions:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        print(func_source)
        print(f"\n--- Generated Tests (Template V1) ---")

        tests = generator.generate(
            func_source,
            prompt_template=PROMPT_TEMPLATE_V1,
            num_return_sequences=3,
            temperature=0.8,
        )

        for i, test in enumerate(tests):
            print(f"\n[Candidate {i+1}]:")
            print(test)

        all_results[func_name] = tests

    return all_results


if __name__ == "__main__":
    import sys

    functions_file = sys.argv[1] if len(sys.argv) > 1 else "functions.py"
    run_baseline_demo(functions_file)
