"""Tests for neural network models."""

import pytest
import torch
import numpy as np

from src.models.lstm_attention import LSTMAttentionEncoder
from src.models.actor_critic import HybridActorCritic


class TestLSTMAttentionEncoder:
    """Test LSTM-Attention encoder."""
    
    @pytest.fixture
    def encoder(self, device):
        return LSTMAttentionEncoder(
            input_dim=5,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        ).to(device)
    
    def test_forward_shape(self, encoder, device, batch_size, sequence_length):
        x = torch.randn(batch_size, sequence_length, 5, device=device)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, 64)
    
    def test_forward_with_hidden(self, encoder, device, batch_size, sequence_length):
        x = torch.randn(batch_size, sequence_length, 5, device=device)
        hidden = encoder.init_hidden(batch_size, device)
        
        output, new_hidden = encoder(x, hidden=hidden, return_hidden=True)
        
        assert output.shape == (batch_size, 64)
        assert new_hidden[0].shape[1] == batch_size
        assert new_hidden[1].shape[1] == batch_size
    
    def test_attention_weights(self, encoder, device, batch_size, sequence_length):
        x = torch.randn(batch_size, sequence_length, 5, device=device)
        
        output, attn_weights = encoder(x, return_attention=True)
        
        assert attn_weights.shape[0] == batch_size
        assert attn_weights.shape[-1] == sequence_length
    
    def test_hidden_state_persistence(self, encoder, device, sequence_length):
        batch_size = 1
        x1 = torch.randn(batch_size, sequence_length, 5, device=device)
        x2 = torch.randn(batch_size, sequence_length, 5, device=device)
        
        # First pass
        hidden = encoder.init_hidden(batch_size, device)
        out1, hidden = encoder(x1, hidden=hidden, return_hidden=True)
        
        # Second pass with persisted hidden state
        out2, _ = encoder(x2, hidden=hidden, return_hidden=True)
        
        # Outputs should be different (hidden state affects output)
        # This tests that hidden state is properly passed through
        assert not torch.allclose(out1, out2)


class TestHybridActorCritic:
    """Test hybrid actor-critic network."""
    
    @pytest.fixture
    def model(self, device, sequence_length, hidden_size):
        return HybridActorCritic(
            sequence_length=sequence_length,
            input_dim=5,
            embed_dim=32,
            hidden_dim=hidden_size,
            num_layers=2,
            num_heads=4,
            num_actions=4,
            prediction_dim=5,
            device=str(device),
        )
    
    def test_forward_shape(self, model, device, batch_size, sequence_length):
        candles = torch.randn(batch_size, sequence_length, 5, device=device)
        position_info = torch.randn(batch_size, 3, device=device)
        account_info = torch.randn(batch_size, 2, device=device)
        
        output = model(candles, position_info, account_info)
        
        assert output.prediction_mean.shape == (batch_size, 5)
        assert output.prediction_std.shape == (batch_size, 5)
        assert output.action_logits.shape == (batch_size, 4)
        assert output.value.shape == (batch_size, 1)
    
    def test_get_action(self, model, device, sequence_length):
        batch_size = 1
        candles = torch.randn(batch_size, sequence_length, 5, device=device)
        position_info = torch.randn(batch_size, 3, device=device)
        account_info = torch.randn(batch_size, 2, device=device)
        
        action, hidden = model.get_action(candles, position_info, account_info)
        
        assert action.prediction.shape == (batch_size, 5)
        assert action.trading_action.shape == (batch_size,)
        assert action.value.shape == (batch_size,)
    
    def test_deterministic_action(self, model, device, sequence_length):
        batch_size = 1
        candles = torch.randn(batch_size, sequence_length, 5, device=device)
        position_info = torch.randn(batch_size, 3, device=device)
        account_info = torch.randn(batch_size, 2, device=device)
        
        action1, _ = model.get_action(candles, position_info, account_info, deterministic=True)
        action2, _ = model.get_action(candles, position_info, account_info, deterministic=True)
        
        # Deterministic actions should be the same
        assert torch.allclose(action1.prediction, action2.prediction)
        assert torch.equal(action1.trading_action, action2.trading_action)
    
    def test_evaluate_actions(self, model, device, batch_size, sequence_length):
        candles = torch.randn(batch_size, sequence_length, 5, device=device)
        position_info = torch.randn(batch_size, 3, device=device)
        account_info = torch.randn(batch_size, 2, device=device)
        predictions = torch.randn(batch_size, 5, device=device)
        trading_actions = torch.randint(0, 4, (batch_size,), device=device)
        
        pred_log_probs, action_log_probs, values, entropy = model.evaluate_actions(
            candles, position_info, account_info, predictions, trading_actions
        )
        
        assert pred_log_probs.shape == (batch_size,)
        assert action_log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
    
    def test_gradient_flow(self, model, device, batch_size, sequence_length):
        candles = torch.randn(batch_size, sequence_length, 5, device=device, requires_grad=True)
        position_info = torch.randn(batch_size, 3, device=device)
        account_info = torch.randn(batch_size, 2, device=device)
        
        output = model(candles, position_info, account_info)
        
        # Check gradient flow
        loss = output.value.sum() + output.action_logits.sum() + output.prediction_mean.sum()
        loss.backward()
        
        assert candles.grad is not None
        assert not torch.isnan(candles.grad).any()

