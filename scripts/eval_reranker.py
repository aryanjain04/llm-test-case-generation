"""
Evaluate critic-based reranking against execution oracle best-of-k.

For each function:
1. Generate K candidates with CodeT5.
2. Score candidates with critic using static feature vectors.
3. Select top-1 by critic score.
4. Also select top-1 by true execution reward (oracle best-of-k).
5. Evaluate both selected sets with the same evaluator.

Example:
python scripts/eval_reranker.py \
  --functions-file functions.py \
  --actor-checkpoint checkpoints/codet5-finetuned/best \
  --critic-checkpoint checkpoints/critics/kan_critic.pt \
  --critic-meta checkpoints/critics/kan_critic_meta.json \
  --critic-type kan \
  --num-candidates 5
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.codet5_generator import CodeT5Generator, PROMPT_TEMPLATE_FINETUNE
from src.execution.sandbox import TestExecutor
from src.evaluation.metrics import Evaluator
from src.ast_analysis.feature_extractor import extract_functions_from_source
from src.rl.critic import CriticFactory
from src.rl.reranker_features import build_combined_feature_vector


def load_critic(checkpoint_path: str, meta_path: str, critic_type: str, device: str):
    payload = torch.load(checkpoint_path, map_location="cpu")

    input_dim = int(payload.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"Invalid input_dim in {checkpoint_path}")

    if critic_type == "kan":
        # pykan is more stable on CPU for many Kaggle/PyTorch combinations.
        critic_device = "cpu"
        model = CriticFactory.create("kan", input_dim=input_dim, device=critic_device).to(critic_device)
    else:
        critic_device = device
        model = CriticFactory.create("mlp", input_dim=input_dim).to(critic_device)

    # Initialize lazy KAN before load_state_dict if needed.
    if len(list(model.parameters())) == 0:
        _ = model(torch.zeros((1, input_dim), dtype=torch.float32, device=critic_device))

    model.load_state_dict(payload["state_dict"])
    model.eval()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mean = meta["mean"]
    std = meta["std"]

    return model, critic_device, mean, std


def normalize(vec: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(x - m) / s for x, m, s in zip(vec, mean, std)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-file", default="functions.py")
    parser.add_argument("--actor-checkpoint", default="checkpoints/codet5-finetuned/best")
    parser.add_argument("--base-model", default="Salesforce/codet5-base")
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--critic-meta", required=True)
    parser.add_argument("--critic-type", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/reranker_eval.json")
    parser.add_argument("--oracle-output", default="results/oracle_bestofk_eval.json")
    parser.add_argument("--details-output", default="results/reranker_details.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load functions to evaluate
    with open(args.functions_file, "r", encoding="utf-8") as f:
        src = f.read()
    functions = extract_functions_from_source(src)
    print(f"Loaded {len(functions)} functions from {args.functions_file}")

    # Load actor and critic
    actor = CodeT5Generator.from_checkpoint(args.actor_checkpoint, base_model=args.base_model, device=device)
    critic, critic_device, mean, std = load_critic(
        checkpoint_path=args.critic_checkpoint,
        meta_path=args.critic_meta,
        critic_type=args.critic_type,
        device=device,
    )

    executor = TestExecutor(timeout=30)
    evaluator = Evaluator()

    selected_by_critic: list[tuple[str, str]] = []
    selected_by_oracle: list[tuple[str, str]] = []
    details = []

    for idx, (func_name, func_code) in enumerate(functions, start=1):
        print(f"[{idx}/{len(functions)}] {func_name}")

        candidates = actor.generate(
            func_code,
            prompt_template=PROMPT_TEMPLATE_FINETUNE,
            num_return_sequences=args.num_candidates,
            temperature=args.temperature,
            do_sample=True,
        )

        scored = []
        for c_idx, cand in enumerate(candidates, start=1):
            vec, _ = build_combined_feature_vector(func_code, cand)
            vec = normalize(vec, mean, std)
            x = torch.tensor([vec], dtype=torch.float32, device=critic_device)

            with torch.no_grad():
                pred = critic(x).item()

            exec_result = executor.execute(func_code, cand)
            scored.append(
                {
                    "candidate_id": c_idx,
                    "pred_score": float(pred),
                    "true_reward": float(exec_result.reward),
                    "pass_rate": float(exec_result.pass_rate),
                    "line_coverage": float(exec_result.line_coverage),
                    "branch_coverage": float(exec_result.branch_coverage),
                    "compilable": bool(exec_result.compilable),
                    "test_code": cand,
                }
            )

        best_pred = max(scored, key=lambda r: r["pred_score"])
        best_true = max(scored, key=lambda r: r["true_reward"])

        selected_by_critic.append((func_name, best_pred["test_code"]))
        selected_by_oracle.append((func_name, best_true["test_code"]))

        details.append(
            {
                "function_name": func_name,
                "best_by_critic": {
                    "candidate_id": best_pred["candidate_id"],
                    "pred_score": best_pred["pred_score"],
                    "true_reward": best_pred["true_reward"],
                },
                "best_by_oracle": {
                    "candidate_id": best_true["candidate_id"],
                    "pred_score": best_true["pred_score"],
                    "true_reward": best_true["true_reward"],
                },
                "candidates": [
                    {
                        "candidate_id": c["candidate_id"],
                        "pred_score": c["pred_score"],
                        "true_reward": c["true_reward"],
                        "pass_rate": c["pass_rate"],
                        "line_coverage": c["line_coverage"],
                        "compilable": c["compilable"],
                    }
                    for c in scored
                ],
            }
        )

    critic_eval = evaluator.evaluate(functions=functions, generated_tests=selected_by_critic)
    oracle_eval = evaluator.evaluate(functions=functions, generated_tests=selected_by_oracle)

    evaluator.save_results(critic_eval, args.output)
    evaluator.save_results(oracle_eval, args.oracle_output)

    details_path = Path(args.details_output)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "critic_type": args.critic_type,
                "critic_checkpoint": args.critic_checkpoint,
                "critic_meta": args.critic_meta,
                "num_candidates": args.num_candidates,
                "details": details,
            },
            f,
            indent=2,
        )

    print("\n=== Critic-Reranker Evaluation ===")
    print("Critic-selected summary:\n" + critic_eval.summary())
    print("Oracle best-of-k summary:\n" + oracle_eval.summary())
    print(f"Saved details to {details_path}")


if __name__ == "__main__":
    main()
