# KAN-Critic: RL-based Unit Test Generation with KAN Value Network

> **Research project** for ICSE submission.  
> Generates Python unit tests using a fine-tuned CodeT5 model (actor) refined with PPO, where a Kolmogorov-Arnold Network (KAN) serves as the critic/value network.

---

## What This Project Does

Given a Python function like this:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Negative input")
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

The system **automatically generates** pytest test cases:

```python
def test_factorial_zero():
    assert factorial(0) == 1

def test_factorial_positive():
    assert factorial(5) == 120

def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)
```

---

## Architecture

```
Source Code
    │
    ▼
AST Parsing + Code Feature Extraction (13-dim vector)
    │
    ▼
Fine-tuned CodeT5 (Actor / Policy Network)
    │  generates candidate test cases
    ▼
Execute Tests → Collect Coverage + Pass Rate (Reward Signal)
    │
    ▼
KAN-based Critic (Value Network)
    │  learns: code features → test quality mapping
    ▼
PPO Update: refine CodeT5 using KAN critic
    │
    ▼
Output: Optimized Test Suite
```

### Why KAN for the Critic?

| Aspect | MLP Critic (baseline) | KAN Critic (ours) |
|--------|----------------------|-------------------|
| Learns | Fixed activations + weights | Activation functions on edges |
| Interpretability | Black box | Can inspect learned functions per feature |
| Suited for | General approximation | Continuous function learning from structured features |

The value function maps **code features → expected test quality** — this is a continuous function approximation task, exactly what KAN excels at.

---

## Project Structure

```
├── functions.py                    # 10 sample functions (original test subjects)
├── test_llm-generated.py           # Hand-written reference tests
├── requirements.txt                # All dependencies
├── configs/
│   └── default.json                # Hyperparameters and paths
├── scripts/
│   ├── run_baseline.py             # Run zero-shot / fine-tuned inference + evaluation
│   └── build_dataset.py            # Mine GitHub repos for function-test pairs
├── src/
│   ├── ast_analysis/
│   │   └── feature_extractor.py    # AST → 13-dim feature vector
│   ├── model/
│   │   ├── codet5_generator.py     # CodeT5 inference wrapper
│   │   └── finetune.py             # LoRA fine-tuning script
│   ├── execution/
│   │   └── sandbox.py              # Sandboxed test runner with coverage
│   ├── data/
│   │   └── dataset_builder.py      # GitHub mining + dataset construction
│   ├── evaluation/
│   │   └── metrics.py              # BLEU, coverage, pass rate, reward
│   └── rl/
│       ├── critic.py               # KAN + MLP critic networks
│       └── ppo_trainer.py          # PPO actor-critic training loop
├── datasets/                       # Training data (JSONL)
├── checkpoints/                    # Saved model weights
└── results/                        # Evaluation outputs
```

---

## Two-Phase Plan

### Phase 1 — Mid-Term (Supervised Baseline)

| Step | What | Status |
|------|------|--------|
| 1 | AST feature extraction working on 10 functions | ✅ Done |
| 2 | Test execution sandbox with coverage measurement | ✅ Done |
| 3 | Mine GitHub repos → build training dataset | 🔧 Ready to run |
| 4 | Fine-tune CodeT5-small with LoRA | 🔧 Ready to run |
| 5 | Evaluate: compilation rate, pass rate, coverage, BLEU | 🔧 Ready to run |
| 6 | Compare zero-shot vs fine-tuned CodeT5 | 🔧 Ready to run |

**Mid-term deliverable:** Fine-tuned CodeT5 that generates compilable, passing tests with measurable coverage — beating the zero-shot baseline.

### Phase 2 — End-Term (RL + KAN)

| Step | What |
|------|------|
| 7 | Integrate KAN critic (install pykan) |
| 8 | Wire PPO training loop: CodeT5 actor + KAN critic |
| 9 | Train with coverage/pass-rate reward signal |
| 10 | Compare: KAN critic vs MLP critic (ablation) |
| 11 | Interpretability analysis of KAN-learned functions |
| 12 | Evaluate on HumanEval / MBPP benchmarks |

**End-term deliverable:** Full KAN-Critic pipeline with ablation study showing KAN's advantage as a value network for code-related RL tasks.

---

## Quick Start

```bash
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test AST feature extraction
python -m src.ast_analysis.feature_extractor functions.py

# 4. Test execution sandbox
python -m src.execution.sandbox

# 5. Test critic networks
python -m src.rl.critic

# 6. Build dataset (mines GitHub repos)
python scripts/build_dataset.py --max-repos 5

# 7. Fine-tune CodeT5
python -m src.model.finetune --data datasets/train.jsonl --epochs 5

# 8. Run baseline evaluation
python scripts/run_baseline.py
```

---

## Code Features Extracted (13 dimensions)

These AST-derived features serve as input to the KAN critic:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `loc` | Lines of code |
| 2 | `param_count` | Number of parameters |
| 3 | `return_count` | Number of return statements |
| 4 | `branch_count` | if/elif/else branches |
| 5 | `loop_count` | for + while loops |
| 6 | `max_nesting_depth` | Deepest nesting level |
| 7 | `has_exception_handling` | try/except present |
| 8 | `is_recursive` | Function calls itself |
| 9 | `function_call_count` | Total function/method calls |
| 10 | `has_default_params` | Any parameter has a default |
| 11 | `cyclomatic_complexity` | McCabe complexity (via radon) |
| 12 | `maintainability_index` | Radon MI score |
| 13 | `assertion_count` | Number of assert statements |

---

## Reward Function

The RL reward signal combines:

```
reward = 0.4 × line_coverage + 0.4 × pass_rate + 0.2 × branch_coverage
```

Returns `-1.0` if generated code doesn't compile, `-0.5` if no tests are found.