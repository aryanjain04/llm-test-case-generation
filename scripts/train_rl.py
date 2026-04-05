"""
Train PPO with either an MLP or KAN critic using a fine-tuned CodeT5 actor.

This script is designed to run on Kaggle after uploading:
1. datasets/train_combined.jsonl
2. the supervised checkpoint folder (best LoRA adapter)

Examples:
    python scripts/train_rl.py --config configs/rl_mlp.json
    python scripts/train_rl.py --config configs/rl_kan.json
"""

import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import jsonlines
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.ppo_trainer import PPOConfig, PPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_function_pool(data_path: str, max_functions: int, seed: int) -> list[str]:
    functions = []
    seen = set()

    with jsonlines.open(data_path) as reader:
        for row in reader:
            code = row.get("function_code", "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            functions.append(code)

    rng = random.Random(seed)
    rng.shuffle(functions)

    if max_functions > 0:
        functions = functions[:max_functions]

    return functions


def load_actor(base_model: str, adapter_checkpoint: str, device: str):
    """Load base model + LoRA adapter in trainable mode."""
    print(f"Loading base model: {base_model}")
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading adapter: {adapter_checkpoint}")
    actor = PeftModel.from_pretrained(base, adapter_checkpoint, is_trainable=True)
    actor.to(device)
    actor.train()

    return actor, tokenizer


def build_ppo_config(cfg: dict) -> PPOConfig:
    p = cfg["ppo"]
    return PPOConfig(
        clip_epsilon=p.get("clip_epsilon", 0.2),
        value_loss_coef=p.get("value_loss_coef", 0.5),
        entropy_coef=p.get("entropy_coef", 0.01),
        gamma=p.get("gamma", 0.99),
        gae_lambda=p.get("gae_lambda", 0.95),
        max_grad_norm=p.get("max_grad_norm", 0.5),
        ppo_epochs=p.get("ppo_epochs", 4),
        batch_size=p.get("batch_size", 4),
        learning_rate_actor=p.get("learning_rate_actor", 1e-5),
        learning_rate_critic=p.get("learning_rate_critic", 3e-4),
        critic_type=p.get("critic_type", "mlp"),
        num_episodes=p.get("num_episodes", 1000),
        log_interval=p.get("log_interval", 10),
    )


def save_json(path: str, payload: dict | list) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_mlp.json")
    parser.add_argument("--critic-type", choices=["mlp", "kan"], default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-functions", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = args.device or cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    base_model = cfg["actor"]["base_model"]
    adapter_checkpoint = cfg["actor"]["adapter_checkpoint"]
    data_path = cfg["data"]["train_jsonl"]
    max_functions = args.max_functions if args.max_functions is not None else int(cfg["data"].get("max_functions", 1000))

    ppo_cfg = build_ppo_config(cfg)
    if args.critic_type:
        ppo_cfg.critic_type = args.critic_type
    if args.num_episodes is not None:
        ppo_cfg.num_episodes = args.num_episodes

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment: {cfg.get('experiment', 'ppo_experiment')}")
    print(f"Critic: {ppo_cfg.critic_type}")
    print(f"Device: {device}")
    print(f"Episodes: {ppo_cfg.num_episodes}")
    print(f"Data: {data_path}")
    print("=" * 60)

    functions = load_function_pool(data_path, max_functions=max_functions, seed=seed)
    if not functions:
        raise ValueError(f"No functions loaded from {data_path}")
    print(f"Loaded {len(functions)} unique functions for RL training")

    actor, tokenizer = load_actor(base_model, adapter_checkpoint, device)

    trainer = PPOTrainer(
        actor_model=actor,
        actor_tokenizer=tokenizer,
        config=ppo_cfg,
        device=device,
    )

    history = trainer.train(functions)

    # Save artifacts
    actor_out = output_dir / "actor_adapter"
    actor.save_pretrained(actor_out)
    tokenizer.save_pretrained(actor_out)

    critic_out = output_dir / "critic.pt"
    torch.save(trainer.critic.state_dict(), critic_out)

    save_json(
        cfg["output"]["train_log"],
        {
            "experiment": cfg.get("experiment", "ppo_experiment"),
            "critic_type": ppo_cfg.critic_type,
            "num_episodes": ppo_cfg.num_episodes,
            "history": history,
            "final_avg_reward": float(np.mean(trainer.episode_rewards[-50:])) if trainer.episode_rewards else 0.0,
            "final_avg_line_coverage": float(np.mean(trainer.episode_coverages[-50:])) if trainer.episode_coverages else 0.0,
        },
    )

    save_json(cfg["output"]["config_dump"], cfg)

    print("\nSaved artifacts:")
    print(f"  Actor adapter: {actor_out}")
    print(f"  Critic state:  {critic_out}")
    print(f"  Train log:     {cfg['output']['train_log']}")


if __name__ == "__main__":
    main()
