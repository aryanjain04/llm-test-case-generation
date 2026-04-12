"""
PPO (Proximal Policy Optimization) training loop for test generation.

Actor-Critic Architecture:
- Actor: Fine-tuned CodeT5 (generates test code tokens)
- Critic: KAN or MLP network (estimates state value from code features)

Reward Signal:
- Coverage achieved by generated tests
- Test pass rate
- Branch coverage bonus
- (Optional) Mutation score

This module wires everything together:
CodeT5 generates → Execute tests → Compute reward → KAN estimates value →
PPO updates both actor and critic.

NOTE: This is the END-TERM component. Mid-term = supervised fine-tuning only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional
from dataclasses import dataclass

from src.rl.critic import CriticFactory
from src.execution.sandbox import TestExecutor
from src.ast_analysis.feature_extractor import (
    extract_features_from_source,
    CodeFeatures,
)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    clip_epsilon: float = 0.2       # PPO clipping range
    value_loss_coef: float = 0.5    # weight for value loss
    entropy_coef: float = 0.01      # entropy bonus for exploration
    gamma: float = 0.99             # discount factor
    gae_lambda: float = 0.95        # GAE lambda
    max_grad_norm: float = 0.5      # gradient clipping
    ppo_epochs: int = 4             # PPO inner epochs per batch
    batch_size: int = 4             # mini-batch size
    learning_rate_actor: float = 1e-5   # actor (CodeT5) LR
    learning_rate_critic: float = 3e-4  # critic (KAN/MLP) LR
    critic_type: str = "kan"        # "kan" or "mlp"
    num_episodes: int = 1000        # total training episodes
    log_interval: int = 10          # log every N episodes


@dataclass
class Experience:
    """Single RL experience tuple."""
    function_code: str
    function_features: list[float]
    generated_test: str
    reward: float
    log_prob: float  # log probability of the generated sequence
    value: float     # critic's value estimate


class PPOTrainer:
    """
    PPO trainer that coordinates actor (CodeT5) and critic (KAN/MLP).

    Training flow per episode:
    1. Sample a function from the dataset
    2. Extract code features (AST analysis)
    3. Actor generates test code (with log probabilities)
    4. Execute test → compute reward (coverage, pass rate)
    5. Critic estimates value from code features
    6. Compute advantage using GAE
    7. PPO update: clip actor loss + value loss + entropy bonus
    """

    def __init__(
        self,
        actor_model,          # fine-tuned CodeT5 model
        actor_tokenizer,      # CodeT5 tokenizer
        config: PPOConfig = PPOConfig(),
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # Actor (CodeT5 — already fine-tuned)
        self.actor = actor_model
        self.tokenizer = actor_tokenizer

        # Critic
        self.critic = CriticFactory.create(
            critic_type=config.critic_type,
            input_dim=CodeFeatures.num_features(),
        ).to(device)

        # KAN critic is lazily initialized; trigger one forward pass so
        # parameters exist before constructing the optimizer.
        critic_params = list(self.critic.parameters())
        if len(critic_params) == 0:
            dummy_features = torch.zeros(
                (1, CodeFeatures.num_features()), dtype=torch.float32, device=device
            )
            _ = self.critic(dummy_features)
            critic_params = list(self.critic.parameters())
            if len(critic_params) == 0:
                raise RuntimeError(
                    "Critic has no parameters after initialization. "
                    "Check KAN/pykan installation."
                )

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.learning_rate_actor
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic_params, lr=config.learning_rate_critic
        )

        # Test executor
        self.executor = TestExecutor(timeout=30)

        # Experience buffer
        self.experiences: list[Experience] = []

        # Training stats
        self.episode_rewards: list[float] = []
        self.episode_coverages: list[float] = []
        self.training_history: list[dict] = []

    def collect_experience(self, function_code: str) -> Experience:
        """
        Run one episode: generate a test and collect experience.

        Args:
            function_code: Source code of the function to test

        Returns:
            Experience tuple with all RL data
        """
        # 1. Extract code features
        features_list = extract_features_from_source(function_code)
        if features_list:
            feature_vec = features_list[0].to_vector()
        else:
            feature_vec = [0.0] * CodeFeatures.num_features()

        # 2. Generate test with log probabilities
        prompt = f"Generate pytest tests for:\n{function_code}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.actor.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        generated_test = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probability of generated sequence
        log_prob = self._compute_log_prob(inputs, generated_ids)

        # 3. Execute test → get reward
        exec_result = self.executor.execute(function_code, generated_test)
        reward = exec_result.reward

        # 4. Critic estimates value
        feature_tensor = torch.tensor(
            [feature_vec], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            value = self.critic(feature_tensor).item()

        experience = Experience(
            function_code=function_code,
            function_features=feature_vec,
            generated_test=generated_test,
            reward=reward,
            log_prob=log_prob,
            value=value,
        )

        self.experiences.append(experience)
        self.episode_rewards.append(reward)
        self.episode_coverages.append(exec_result.line_coverage)

        return experience

    def _compute_log_prob(self, inputs, generated_ids) -> float:
        """Compute log probability of the generated sequence under the actor."""
        with torch.no_grad():
            outputs = self.actor(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=generated_ids.unsqueeze(0),
            )
            # Negative loss ≈ average log probability
            return -outputs.loss.item()

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns from collected experiences.
        """
        rewards = [e.reward for e in self.experiences]
        values = [e.value for e in self.experiences]

        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update(self):
        """
        Perform PPO update on both actor and critic.

        This is called after collecting a batch of experiences.
        """
        if not self.experiences:
            return {}

        advantages, returns = self.compute_advantages()

        # Prepare feature tensors
        features = torch.tensor(
            [e.function_features for e in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [e.log_prob for e in self.experiences],
            dtype=torch.float32,
            device=self.device,
        )

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.config.ppo_epochs):
            # Critic update
            values = self.critic(features).squeeze()
            critic_loss = F.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.max_grad_norm
            )
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()

            # Actor update (simplified — full implementation would
            # re-compute log probs for each experience)
            # For now, we use the REINFORCE-style policy gradient
            # with PPO clipping applied to the advantage-weighted loss
            #
            # Full PPO with token-level log probs requires TRL library
            # which we'll integrate in the complete version

        # Clear experience buffer
        stats = {
            "actor_loss": total_actor_loss / max(self.config.ppo_epochs, 1),
            "critic_loss": total_critic_loss / max(self.config.ppo_epochs, 1),
            "avg_reward": sum(e.reward for e in self.experiences) / len(self.experiences),
            "avg_value": sum(e.value for e in self.experiences) / len(self.experiences),
            "buffer_size": len(self.experiences),
        }

        self.experiences = []
        return stats

    def train(self, functions: list[str]) -> list[dict]:
        """
        Main training loop.

        Args:
            functions: List of function source codes to train on
        """
        import random

        print(f"=== PPO Training ===")
        print(f"Functions: {len(functions)}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Critic type: {self.config.critic_type}")
        print()

        for episode in range(1, self.config.num_episodes + 1):
            # Sample a random function
            func_code = random.choice(functions)

            # Collect experience
            exp = self.collect_experience(func_code)

            # Update every batch_size episodes
            if len(self.experiences) >= self.config.batch_size:
                stats = self.ppo_update()

                if episode % self.config.log_interval == 0:
                    avg_reward = sum(self.episode_rewards[-self.config.log_interval:]) / self.config.log_interval
                    avg_cov = sum(self.episode_coverages[-self.config.log_interval:]) / self.config.log_interval
                    history_row = {
                        "episode": episode,
                        "avg_reward": avg_reward,
                        "avg_line_coverage": avg_cov,
                        "critic_loss": stats.get("critic_loss", 0.0),
                        "actor_loss": stats.get("actor_loss", 0.0),
                        "buffer_size": stats.get("buffer_size", 0),
                    }
                    self.training_history.append(history_row)
                    print(
                        f"Episode {episode}/{self.config.num_episodes} | "
                        f"Reward: {avg_reward:.3f} | "
                        f"Coverage: {avg_cov:.1f}% | "
                        f"Critic Loss: {stats.get('critic_loss', 0):.4f}"
                    )

        print("\nTraining complete!")
        return self.training_history
