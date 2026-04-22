"""
Train a critic model (MLP or KAN) to predict candidate reward.

This is supervised critic training for reranking, independent of end-to-end PPO.

Example:
python scripts/train_critic.py \
  --data datasets/critic_reranker_train.jsonl \
  --critic-type mlp \
  --epochs 80 --batch-size 64 \
  --output checkpoints/critics/mlp_critic.pt \
  --meta-output checkpoints/critics/mlp_critic_meta.json
"""

from __future__ import annotations

import sys
import json
import random
import argparse
from pathlib import Path

import torch
import jsonlines
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.critic import CriticFactory
from src.rl.reranker_features import compute_standardization_stats


class CriticDataset(Dataset):
    def __init__(self, xs: list[list[float]], ys: list[float]):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.xs[idx], dtype=torch.float32),
            torch.tensor(self.ys[idx], dtype=torch.float32),
        )


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: str) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    with jsonlines.open(path) as reader:
        for row in reader:
            vec = row.get("features", None)
            tgt = row.get("target_reward", None)
            if vec is None or tgt is None:
                continue
            xs.append([float(v) for v in vec])
            ys.append(float(tgt))
    return xs, ys


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def select(data: list, idx: list[int]):
    return [data[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/critic_reranker_train.jsonl")
    parser.add_argument("--critic-type", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--target", default="target_reward")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="checkpoints/critics/mlp_critic.pt")
    parser.add_argument("--meta-output", default="checkpoints/critics/mlp_critic_meta.json")
    parser.add_argument("--log-output", default="results/mlp_critic_train_log.json")
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--kan-order", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    xs, ys = load_rows(args.data)
    if not xs:
        raise ValueError(f"No usable rows in {args.data}")

    input_dim = len(xs[0])
    print(f"Loaded {len(xs)} rows, input_dim={input_dim}")

    train_idx, val_idx = split_indices(len(xs), args.val_ratio, args.seed)
    x_train, y_train = select(xs, train_idx), select(ys, train_idx)
    x_val, y_val = select(xs, val_idx), select(ys, val_idx)

    mean, std = compute_standardization_stats(x_train)

    def normalize(vectors: list[list[float]]) -> list[list[float]]:
        out = []
        for vec in vectors:
            out.append([(x - m) / s for x, m, s in zip(vec, mean, std)])
        return out

    x_train = normalize(x_train)
    x_val = normalize(x_val)

    train_ds = CriticDataset(x_train, y_train)
    val_ds = CriticDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # KAN in pykan is more stable on CPU in many environments.
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    critic_device = "cpu" if args.critic_type == "kan" else requested_device

    if args.critic_type == "kan":
        model = CriticFactory.create(
            critic_type="kan",
            input_dim=input_dim,
            grid_size=args.kan_grid,
            spline_order=args.kan_order,
            device=critic_device,
        ).to(critic_device)
    else:
        model = CriticFactory.create(
            critic_type="mlp",
            input_dim=input_dim,
        ).to(critic_device)

    params = list(model.parameters())
    if not params:
        _ = model(torch.zeros((1, input_dim), dtype=torch.float32, device=critic_device))
        params = list(model.parameters())
        if not params:
            raise RuntimeError("Critic has no parameters after initialization")

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    mse = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(critic_device)
            yb = yb.to(critic_device)

            pred = model(xb).squeeze(-1)
            loss = mse(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(critic_device)
                yb = yb.to(critic_device)
                pred = model(xb).squeeze(-1)
                val_loss += mse(pred, yb).item()
        val_loss /= max(len(val_loader), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f}")

    if best_state is None:
        raise RuntimeError("Training finished without best state")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "critic_type": args.critic_type,
            "input_dim": input_dim,
            "state_dict": best_state,
            "best_val_loss": best_val,
            "device_used": critic_device,
        },
        out_path,
    )

    feature_names = []
    with jsonlines.open(args.data) as reader:
        first = next(iter(reader), None)
        if first is not None:
            n = len(first.get("features", []))
            feature_names = [f"f_{i}" for i in range(n)]

    meta = {
        "data": args.data,
        "critic_type": args.critic_type,
        "target": args.target,
        "input_dim": input_dim,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_val_loss": best_val,
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
        "device_used": critic_device,
    }

    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_path = Path(args.log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n=== Done ===")
    print(f"Saved checkpoint: {out_path}")
    print(f"Saved metadata:   {meta_path}")
    print(f"Saved train log:  {log_path}")


if __name__ == "__main__":
    main()
