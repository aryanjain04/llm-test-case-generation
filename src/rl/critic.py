"""
KAN (Kolmogorov-Arnold Network) Critic for RL-based test generation.

This is the VALUE NETWORK in the actor-critic framework.
- Input: code feature vector (from AST analysis) + test quality signals
- Output: estimated value (expected future reward)

WHY KAN here:
1. The value function maps continuous code features → continuous quality score
   This is fundamentally a function approximation task — KAN's sweet spot.
2. KAN's learned activation functions are interpretable — we can analyze
   WHICH code features matter for test quality (this is a paper contribution).
3. For the small feature vectors we use (~13 dims), KAN is efficient and
   trains quickly on CPU/small GPU.

NOTE: pykan must be installed: pip install pykan
This module is for the END-TERM phase. The mid-term uses the standard
CodeT5 baseline without RL.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path


class MLPCritic(nn.Module):
    """
    Standard MLP critic (baseline to compare KAN against).

    This is the CONTROL — if KAN critic doesn't beat this, there's no
    contribution from using KAN.
    """

    def __init__(self, input_dim: int = 13, hidden_dims: list[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # single value output

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim] code feature vectors

        Returns:
            values: [batch_size, 1] estimated state values
        """
        return self.network(features)


class KANCritic(nn.Module):
    """
    KAN-based critic network.

    Architecture: Uses B-spline basis functions on edges instead of
    fixed activations with learned weights. Each edge learns its own
    activation function.

    For the actual KAN implementation, we use the pykan library.
    This wrapper makes it compatible with our PPO training loop.
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: list[int] = [8, 4],
        grid_size: int = 5,
        spline_order: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.grid_size = grid_size
        self.spline_order = spline_order
        self._device = device

        # Build layer dimensions: [input] -> [hidden1] -> [hidden2] -> [1]
        self.layer_dims = [input_dim] + hidden_dims + [1]

        # We'll lazy-init the KAN model to avoid import errors if pykan isn't installed
        self._kan_model = None
        self._initialized = False

    def _init_kan(self):
        """Lazy initialization of KAN model."""
        try:
            from kan import KAN

            self._kan_model = KAN(
                width=self.layer_dims,
                grid=self.grid_size,
                k=self.spline_order,
                device=self._device,
            )
            self._initialized = True
            print(f"KAN critic initialized: {self.layer_dims}")
        except ImportError:
            raise ImportError(
                "pykan is required for KAN critic. Install with: pip install pykan\n"
                "This is needed for the end-term RL phase only."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim] code feature vectors

        Returns:
            values: [batch_size, 1] estimated state values
        """
        if not self._initialized:
            self._init_kan()

        return self._kan_model(features)

    def get_interpretability_report(self) -> dict:
        """
        Extract interpretability information from learned KAN activations.

        This is a KEY CONTRIBUTION — examining what the KAN learned about
        which code features predict test quality.
        """
        if not self._initialized:
            return {"error": "Model not initialized"}

        # TODO: Extract and visualize learned activation functions per edge
        # The pykan library provides .plot() for visualization
        # We'll structure this as a dict mapping feature names → importance scores
        report = {
            "layer_dims": self.layer_dims,
            "grid_size": self.grid_size,
            "note": "Full interpretability analysis to be implemented after training",
        }
        return report


class CriticFactory:
    """Factory to create the right critic based on configuration."""

    @staticmethod
    def create(
        critic_type: str = "mlp",
        input_dim: int = 13,
        **kwargs,
    ) -> nn.Module:
        """
        Create a critic network.

        Args:
            critic_type: "mlp" or "kan"
            input_dim: Number of input features
            **kwargs: Additional arguments for the specific critic type

        Returns:
            nn.Module critic network
        """
        if critic_type == "mlp":
            return MLPCritic(input_dim=input_dim, **kwargs)
        elif critic_type == "kan":
            return KANCritic(input_dim=input_dim, **kwargs)
        else:
            raise ValueError(f"Unknown critic type: {critic_type}. Use 'mlp' or 'kan'.")


# --- Quick test ---
if __name__ == "__main__":
    print("=== Testing MLP Critic ===")
    mlp = MLPCritic(input_dim=13)
    dummy_features = torch.randn(4, 13)  # batch of 4
    values = mlp(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {values.shape}")
    print(f"Values: {values.squeeze().tolist()}")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")

    print("\n=== Testing KAN Critic ===")
    try:
        kan = KANCritic(input_dim=13, device="cpu")
        values = kan(dummy_features)
        print(f"KAN output shape: {values.shape}")
        print(f"KAN values: {values.squeeze().tolist()}")
    except ImportError as e:
        print(f"KAN not available (expected for mid-term): {e}")

    print("\n=== Factory Test ===")
    critic = CriticFactory.create("mlp", input_dim=13)
    print(f"Created: {type(critic).__name__}")
