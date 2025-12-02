"""Hybrid Actor-Critic network for PPO with continuous and discrete actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.models.components import MLP, get_device
from src.models.lstm_attention import LSTMAttentionEncoder
from src.config.constants import (
    NUM_OHLCV_FEATURES,
    NUM_POSITION_FEATURES,
    NUM_ACCOUNT_FEATURES,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_SEQUENCE_LENGTH,
)


class ActionOutput(NamedTuple):
    """Output of the actor network."""
    
    # Continuous action (predicted candle)
    prediction_mean: torch.Tensor  # [batch, 5]
    prediction_std: torch.Tensor   # [batch, 5]
    
    # Discrete action (trading decision)
    action_logits: torch.Tensor    # [batch, 4]
    
    # Value estimate
    value: torch.Tensor            # [batch, 1]


@dataclass
class HybridAction:
    """Sampled hybrid action."""
    
    prediction: torch.Tensor  # [batch, 5] - predicted OHLCV
    trading_action: torch.Tensor  # [batch] - discrete action index
    prediction_log_prob: torch.Tensor  # [batch]
    action_log_prob: torch.Tensor  # [batch]
    value: torch.Tensor  # [batch]


class HybridActorCritic(nn.Module):
    """Hybrid Actor-Critic network for PPO.
    
    This network processes candle sequences and outputs:
    1. Continuous prediction: Gaussian distribution over next candle (OHLCV)
    2. Discrete action: Categorical distribution over trading actions
    3. Value: State value estimate for PPO
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     LSTM-Attention Encoder                      │
    │  Input: [batch, 120, 5] -> Output: [batch, hidden_dim]         │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Prediction Head │  │   Action Head   │  │   Value Head    │
    │  (Continuous)    │  │   (Discrete)    │  │   (Critic)      │
    │  μ, σ for OHLCV  │  │  4 action logits│  │  scalar value   │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        input_dim: int = NUM_OHLCV_FEATURES,
        embed_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_heads: int = DEFAULT_ATTENTION_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        num_actions: int = 4,
        prediction_dim: int = 5,
        min_std: float = 0.01,
        max_std: float = 1.0,
        device: str = "auto",
    ) -> None:
        """Initialize the Hybrid Actor-Critic network.
        
        Args:
            sequence_length: Number of candles in input sequence
            input_dim: Number of features per candle
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_actions: Number of discrete actions
            prediction_dim: Dimension of prediction output
            min_std: Minimum standard deviation for predictions
            max_std: Maximum standard deviation for predictions
            device: Device to use
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.prediction_dim = prediction_dim
        self.min_std = min_std
        self.max_std = max_std
        self._device = get_device(device)
        
        # Shared encoder
        self.encoder = LSTMAttentionEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Position and account info embedding
        self.position_embed = nn.Sequential(
            nn.Linear(NUM_POSITION_FEATURES, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        self.account_embed = nn.Sequential(
            nn.Linear(NUM_ACCOUNT_FEATURES, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        
        # Combined feature dimension
        combined_dim = hidden_dim + 32 + 32  # encoder + position + account
        
        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Prediction head (continuous) - outputs mean and log_std
        self.prediction_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[128, 64],
            output_dim=prediction_dim * 2,  # mean and log_std
            activation="relu",
            dropout=dropout,
        )
        
        # Action head (discrete)
        self.action_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[128, 64],
            output_dim=num_actions,
            activation="relu",
            dropout=dropout,
        )
        
        # Value head (critic)
        self.value_head = MLP(
            input_dim=hidden_dim,
            hidden_dims=[128, 64],
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )
        
        # Move to device
        self.to(self._device)

    def forward(
        self,
        candles: torch.Tensor,
        position_info: torch.Tensor,
        account_info: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False,
    ) -> ActionOutput | tuple[ActionOutput, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.
        
        Args:
            candles: Candle sequence [batch, seq_len, 5]
            position_info: Position features [batch, 3]
            account_info: Account features [batch, 2]
            hidden: Optional LSTM hidden state
            return_hidden: Whether to return new hidden state
            
        Returns:
            ActionOutput with distributions and value, optionally with hidden state
        """
        # Encode candle sequence
        if return_hidden:
            encoded, new_hidden = self.encoder(
                candles, hidden=hidden, return_hidden=True
            )
        else:
            encoded = self.encoder(candles, hidden=hidden)
            new_hidden = None
        
        # Embed position and account info
        pos_embedded = self.position_embed(position_info)
        acc_embedded = self.account_embed(account_info)
        
        # Combine features
        combined = torch.cat([encoded, pos_embedded, acc_embedded], dim=-1)
        features = self.feature_combiner(combined)
        
        # Get outputs from each head
        pred_out = self.prediction_head(features)
        pred_mean = pred_out[:, :self.prediction_dim]
        pred_log_std = pred_out[:, self.prediction_dim:]
        
        # Constrain std to reasonable range
        pred_std = torch.clamp(
            torch.exp(pred_log_std),
            self.min_std,
            self.max_std,
        )
        
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        output = ActionOutput(
            prediction_mean=pred_mean,
            prediction_std=pred_std,
            action_logits=action_logits,
            value=value,
        )
        
        if return_hidden and new_hidden is not None:
            return output, new_hidden
        return output

    def get_action(
        self,
        candles: torch.Tensor,
        position_info: torch.Tensor,
        account_info: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[HybridAction, tuple[torch.Tensor, torch.Tensor] | None]:
        """Sample an action from the policy.
        
        Args:
            candles: Candle sequence [batch, seq_len, 5]
            position_info: Position features [batch, 3]
            account_info: Account features [batch, 2]
            hidden: Optional LSTM hidden state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (HybridAction, new_hidden_state or None)
        """
        output, new_hidden = self.forward(
            candles, position_info, account_info,
            hidden=hidden, return_hidden=True,
        )
        
        # Sample or select prediction
        if deterministic:
            prediction = output.prediction_mean
            pred_log_prob = torch.zeros(candles.shape[0], device=candles.device)
        else:
            pred_dist = Normal(output.prediction_mean, output.prediction_std)
            prediction = pred_dist.sample()
            pred_log_prob = pred_dist.log_prob(prediction).sum(dim=-1)
        
        # Sample or select discrete action
        if deterministic:
            trading_action = output.action_logits.argmax(dim=-1)
            action_log_prob = torch.zeros(candles.shape[0], device=candles.device)
        else:
            action_dist = Categorical(logits=output.action_logits)
            trading_action = action_dist.sample()
            action_log_prob = action_dist.log_prob(trading_action)
        
        hybrid_action = HybridAction(
            prediction=prediction,
            trading_action=trading_action,
            prediction_log_prob=pred_log_prob,
            action_log_prob=action_log_prob,
            value=output.value.squeeze(-1),
        )
        
        return hybrid_action, new_hidden

    def evaluate_actions(
        self,
        candles: torch.Tensor,
        position_info: torch.Tensor,
        account_info: torch.Tensor,
        predictions: torch.Tensor,
        trading_actions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.
        
        Args:
            candles: Candle sequences [batch, seq_len, 5]
            position_info: Position features [batch, 3]
            account_info: Account features [batch, 2]
            predictions: Taken predictions [batch, 5]
            trading_actions: Taken discrete actions [batch]
            hidden: Optional LSTM hidden state
            
        Returns:
            Tuple of:
            - prediction_log_probs: Log prob of predictions [batch]
            - action_log_probs: Log prob of discrete actions [batch]
            - values: State values [batch]
            - entropy: Combined entropy for exploration [batch]
        """
        output = self.forward(candles, position_info, account_info, hidden=hidden)
        
        # Prediction distribution
        pred_dist = Normal(output.prediction_mean, output.prediction_std)
        pred_log_prob = pred_dist.log_prob(predictions).sum(dim=-1)
        pred_entropy = pred_dist.entropy().sum(dim=-1)
        
        # Action distribution
        action_dist = Categorical(logits=output.action_logits)
        action_log_prob = action_dist.log_prob(trading_actions)
        action_entropy = action_dist.entropy()
        
        # Combined entropy (weighted)
        entropy = action_entropy + 0.1 * pred_entropy
        
        return (
            pred_log_prob,
            action_log_prob,
            output.value.squeeze(-1),
            entropy,
        )

    def get_value(
        self,
        candles: torch.Tensor,
        position_info: torch.Tensor,
        account_info: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get state value estimate only.
        
        Args:
            candles: Candle sequence [batch, seq_len, 5]
            position_info: Position features [batch, 3]
            account_info: Account features [batch, 2]
            hidden: Optional LSTM hidden state
            
        Returns:
            State value [batch]
        """
        output = self.forward(candles, position_info, account_info, hidden=hidden)
        return output.value.squeeze(-1)

    def init_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (h_0, c_0)
        """
        return self.encoder.init_hidden(batch_size, device=self._device)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=self._device))

    @property
    def device(self) -> torch.device:
        """Get the model device."""
        return self._device

