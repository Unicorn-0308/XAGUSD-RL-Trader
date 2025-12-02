"""PPO Agent implementation for hybrid action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.models.actor_critic import HybridActorCritic, HybridAction
from src.agent.rollout_buffer import RolloutBuffer, RolloutBatch
from src.config.constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_CLIP_EPSILON,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_VALUE_COEF,
    DEFAULT_PREDICTION_COEF,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_BATCH_SIZE,
)


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    
    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    value_coef: float = DEFAULT_VALUE_COEF
    prediction_coef: float = DEFAULT_PREDICTION_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize_advantages: bool = True
    target_kl: float | None = 0.02  # Early stopping if KL divergence exceeds this


@dataclass
class UpdateResult:
    """Results from a PPO update."""
    
    policy_loss: float
    value_loss: float
    prediction_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    epochs_completed: int


class PPOAgent:
    """Proximal Policy Optimization agent for hybrid action space.
    
    This agent handles:
    - Continuous predictions (next candle OHLCV)
    - Discrete trading actions (NONE, BUY, SELL, CLOSE)
    
    The loss function combines:
    - Clipped policy loss (for discrete actions)
    - Prediction likelihood loss (for continuous predictions)
    - Value function loss
    - Entropy bonus
    """
    
    def __init__(
        self,
        model: HybridActorCritic,
        config: PPOConfig | None = None,
    ) -> None:
        """Initialize the PPO agent.
        
        Args:
            model: The actor-critic network
            config: PPO configuration
        """
        self.model = model
        self.config = config or PPOConfig()
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        
        # LSTM hidden state management
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    
    @torch.no_grad()
    def get_action(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> tuple[HybridAction, dict[str, Any]]:
        """Get action for a single observation.
        
        Args:
            observation: Environment observation dict
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (HybridAction, info dict)
        """
        self.model.eval()
        
        # Convert observation to tensors
        candles = torch.tensor(
            observation["candles"][np.newaxis],  # Add batch dim
            device=self.model.device,
            dtype=torch.float32,
        )
        position_info = torch.tensor(
            observation["position"][np.newaxis],
            device=self.model.device,
            dtype=torch.float32,
        )
        account_info = torch.tensor(
            observation["account"][np.newaxis],
            device=self.model.device,
            dtype=torch.float32,
        )
        
        # Get action from model
        action, new_hidden = self.model.get_action(
            candles=candles,
            position_info=position_info,
            account_info=account_info,
            hidden=self._hidden,
            deterministic=deterministic,
        )
        
        # Update hidden state
        self._hidden = new_hidden
        
        # Convert to numpy for environment
        info = {
            "prediction": action.prediction[0].cpu().numpy(),
            "trading_action": int(action.trading_action[0].cpu().item()),
            "value": float(action.value[0].cpu().item()),
            "pred_log_prob": float(action.prediction_log_prob[0].cpu().item()),
            "action_log_prob": float(action.action_log_prob[0].cpu().item()),
        }
        
        return action, info
    
    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state.
        
        Args:
            batch_size: Batch size for hidden state
        """
        self._hidden = self.model.init_hidden(batch_size)
    
    def detach_hidden(self) -> None:
        """Detach hidden state from computation graph."""
        if self._hidden is not None:
            self._hidden = (
                self._hidden[0].detach(),
                self._hidden[1].detach(),
            )
    
    def update(self, buffer: RolloutBuffer) -> UpdateResult:
        """Perform PPO update using collected rollout data.
        
        Args:
            buffer: Rollout buffer with collected experience
            
        Returns:
            UpdateResult with training metrics
        """
        self.model.train()
        
        # Normalize advantages if configured
        if self.config.normalize_advantages:
            buffer.normalize_advantages()
        
        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_pred_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0
        epochs_completed = 0
        
        # Multiple epochs over the data
        for epoch in range(self.config.num_epochs):
            epoch_kl = 0.0
            epoch_updates = 0
            
            for batch in buffer.get_batches(self.config.batch_size, shuffle=True):
                # Compute losses
                losses = self._compute_losses(batch)
                
                # Update metrics
                total_policy_loss += losses["policy_loss"]
                total_value_loss += losses["value_loss"]
                total_pred_loss += losses["prediction_loss"]
                total_entropy += losses["entropy"]
                total_approx_kl += losses["approx_kl"]
                total_clip_fraction += losses["clip_fraction"]
                epoch_kl += losses["approx_kl"]
                
                # Backward pass
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                self.optimizer.step()
                
                num_updates += 1
                epoch_updates += 1
            
            epochs_completed = epoch + 1
            
            # Early stopping based on KL divergence
            if self.config.target_kl is not None:
                avg_epoch_kl = epoch_kl / max(epoch_updates, 1)
                if avg_epoch_kl > self.config.target_kl:
                    break
        
        # Compute explained variance
        all_data = buffer.get_all()
        explained_var = self._compute_explained_variance(
            all_data.values.cpu().numpy(),
            all_data.returns.cpu().numpy(),
        )
        
        # Average metrics
        num_updates = max(num_updates, 1)
        
        return UpdateResult(
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            prediction_loss=total_pred_loss / num_updates,
            entropy=total_entropy / num_updates,
            total_loss=(total_policy_loss + total_value_loss + total_pred_loss) / num_updates,
            approx_kl=total_approx_kl / num_updates,
            clip_fraction=total_clip_fraction / num_updates,
            explained_variance=explained_var,
            epochs_completed=epochs_completed,
        )
    
    def _compute_losses(self, batch: RolloutBatch) -> dict[str, Any]:
        """Compute all PPO losses for a batch.
        
        Args:
            batch: Rollout batch
            
        Returns:
            Dictionary of losses and metrics
        """
        # Evaluate actions under current policy
        pred_log_probs, action_log_probs, values, entropy = self.model.evaluate_actions(
            candles=batch.candles,
            position_info=batch.position_info,
            account_info=batch.account_info,
            predictions=batch.predictions,
            trading_actions=batch.trading_actions,
        )
        
        # Compute ratios
        action_ratio = torch.exp(action_log_probs - batch.old_action_log_probs)
        pred_ratio = torch.exp(pred_log_probs - batch.old_pred_log_probs)
        
        # Clipped policy loss (for discrete actions)
        advantages = batch.advantages
        surr1 = action_ratio * advantages
        surr2 = torch.clamp(
            action_ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Prediction loss (negative log likelihood, also clipped)
        pred_surr1 = pred_ratio * advantages
        pred_surr2 = torch.clamp(
            pred_ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages
        prediction_loss = -torch.min(pred_surr1, pred_surr2).mean()
        
        # Value loss (clipped)
        value_pred = values
        value_target = batch.returns
        value_loss_unclipped = (value_pred - value_target) ** 2
        
        # Clip value predictions
        value_clipped = batch.values + torch.clamp(
            value_pred - batch.values,
            -self.config.clip_epsilon,
            self.config.clip_epsilon,
        )
        value_loss_clipped = (value_clipped - value_target) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss
            + self.config.prediction_coef * prediction_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )
        
        # Compute metrics
        with torch.no_grad():
            approx_kl = ((action_ratio - 1) - torch.log(action_ratio)).mean().item()
            clip_fraction = (
                (torch.abs(action_ratio - 1.0) > self.config.clip_epsilon)
                .float()
                .mean()
                .item()
            )
        
        return {
            "policy_loss": policy_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
    
    @staticmethod
    def _compute_explained_variance(
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute explained variance.
        
        explained_variance = 1 - Var(y_true - y_pred) / Var(y_true)
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Explained variance (0 to 1, higher is better)
        """
        var_y = np.var(y_true)
        if var_y == 0:
            return 0.0
        return 1 - np.var(y_true - y_pred) / var_y
    
    def save(self, path: str) -> None:
        """Save agent state.
        
        Args:
            path: Path to save to
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state.
        
        Args:
            path: Path to load from
        """
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "config" in checkpoint:
            self.config = checkpoint["config"]
    
    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self.model.device

