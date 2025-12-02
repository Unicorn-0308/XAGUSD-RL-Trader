"""Reusable neural network components."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.
    
    Features:
    - LayerNorm or BatchNorm option
    - Dropout support
    - Configurable activation functions
    - Optional output activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: str | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Sequence of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name (relu, gelu, tanh, leaky_relu)
            output_activation: Optional activation for output layer
            dropout: Dropout probability
            use_layer_norm: Whether to use LayerNorm (vs no normalization)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=-1),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def _init_weights(self) -> None:
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence data.
    
    Computes attention weights over a sequence using a query vector,
    then produces a context vector as a weighted sum of values.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention.
        
        Args:
            query: Query tensor of shape [batch, hidden_dim]
            keys: Key tensor of shape [batch, seq_len, hidden_dim]
            values: Value tensor of shape [batch, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (context vector [batch, hidden_dim], attention weights [batch, num_heads, seq_len])
        """
        batch_size, seq_len, _ = keys.shape
        
        # Project and reshape for multi-head attention
        # Query: [batch, hidden] -> [batch, 1, hidden] -> [batch, num_heads, 1, head_dim]
        q = self.query_proj(query).unsqueeze(1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Keys: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
        k = self.key_proj(keys)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Values: [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
        v = self.value_proj(values)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: [batch, num_heads, 1, seq]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [batch, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape: [batch, hidden]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)
        context = self.out_proj(context)
        
        # Residual connection with query
        context = self.layer_norm(context + query)
        
        # Return attention weights for visualization
        return context, attn_weights.squeeze(2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ) -> None:
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GatedResidualBlock(nn.Module):
    """Gated residual block for better gradient flow."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize gated residual block.
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating mechanism."""
        residual = x
        
        x = self.layer_norm(x)
        h = self.activation(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Gating
        g = torch.sigmoid(self.gate(x))
        
        return residual + g * h


class InputEmbedding(nn.Module):
    """Embedding layer for OHLCV input features."""

    def __init__(
        self,
        input_dim: int = 5,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initialize input embedding.
        
        Args:
            input_dim: Number of input features (5 for OHLCV)
            embed_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.linear = nn.Linear(input_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input features.
        
        Args:
            x: Input of shape [batch, seq_len, input_dim]
            
        Returns:
            Embedded features of shape [batch, seq_len, embed_dim]
        """
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate torch device.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu")
        
    Returns:
        torch.device instance
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize module weights using orthogonal initialization."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)

