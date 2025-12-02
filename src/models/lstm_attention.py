"""LSTM with Attention mechanism for sequence encoding."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components import (
    InputEmbedding,
    TemporalAttention,
    PositionalEncoding,
    init_weights_orthogonal,
)


class LSTMAttentionEncoder(nn.Module):
    """LSTM encoder with temporal attention for candle sequence processing.
    
    Architecture:
    1. Input Embedding: [batch, seq_len, 5] -> [batch, seq_len, embed_dim]
    2. Positional Encoding: Add position information
    3. LSTM: Process sequence, output [batch, seq_len, hidden_dim]
    4. Temporal Attention: Attend to important time steps
    5. Output: Context vector [batch, hidden_dim]
    
    The LSTM hidden state (h_n, c_n) is maintained between calls to preserve
    memory of the entire history, not just the current window.
    """

    def __init__(
        self,
        input_dim: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        """Initialize the LSTM-Attention encoder.
        
        Args:
            input_dim: Number of input features (5 for OHLCV)
            embed_dim: Embedding dimension after input projection
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input embedding
        self.input_embedding = InputEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=embed_dim,
            max_len=500,
            dropout=dropout,
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Adjust for bidirectional
        lstm_output_dim = hidden_dim * self.num_directions
        
        # Layer norm after LSTM
        self.lstm_layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Project bidirectional output to hidden_dim if needed
        if bidirectional:
            self.bidirectional_proj = nn.Linear(lstm_output_dim, hidden_dim)
            attention_dim = hidden_dim
        else:
            self.bidirectional_proj = None
            attention_dim = hidden_dim
        
        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Initialize weights
        self.apply(lambda m: init_weights_orthogonal(m, gain=1.0))

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            hidden: Optional tuple of (h_n, c_n) LSTM hidden states
            return_hidden: Whether to return the new hidden state
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple containing:
            - context: Encoded representation [batch, hidden_dim]
            - hidden (optional): New LSTM hidden state (h_n, c_n)
            - attention_weights (optional): Attention weights [batch, num_heads, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Input embedding: [batch, seq, 5] -> [batch, seq, embed_dim]
        embedded = self.input_embedding(x)
        
        # 2. Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # 3. LSTM encoding
        if hidden is not None:
            lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Layer norm
        lstm_out = self.lstm_layer_norm(lstm_out)
        
        # Project bidirectional output if needed
        if self.bidirectional_proj is not None:
            lstm_out = self.bidirectional_proj(lstm_out)
            # Also project hidden state for consistency
            h_n_combined = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_dim)
            h_n = h_n_combined[:, 0, :, :] + h_n_combined[:, 1, :, :]  # Sum directions
        
        # 4. Get query from last hidden state (last layer)
        query = h_n[-1]  # [batch, hidden_dim]
        
        # 5. Apply temporal attention
        context, attn_weights = self.attention(
            query=query,
            keys=lstm_out,
            values=lstm_out,
        )
        
        # Build return tuple
        outputs = [context]
        
        if return_hidden:
            outputs.append((h_n, c_n))
        
        if return_attention:
            outputs.append(attn_weights)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for the LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        if device is None:
            device = next(self.parameters()).device
        
        num_directions = self.num_directions
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        
        return h_0, c_0

    def detach_hidden(
        self,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detach hidden state from computation graph.
        
        This is useful for truncated BPTT to prevent backprop through
        the entire history.
        
        Args:
            hidden: Tuple of (h_n, c_n) hidden states
            
        Returns:
            Detached hidden states
        """
        h_n, c_n = hidden
        return h_n.detach(), c_n.detach()

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.hidden_dim


class LSTMEncoder(nn.Module):
    """Simpler LSTM encoder without attention (for comparison/ablation)."""

    def __init__(
        self,
        input_dim: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize simple LSTM encoder."""
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_embedding = InputEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass."""
        embedded = self.input_embedding(x)
        
        if hidden is not None:
            lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        else:
            lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use last hidden state as output
        output = self.layer_norm(h_n[-1])
        
        if return_hidden:
            return output, (h_n, c_n)
        return output

    @property
    def output_dim(self) -> int:
        """Get the output dimension."""
        return self.hidden_dim

