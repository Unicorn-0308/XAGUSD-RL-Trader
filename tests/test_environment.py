"""Tests for the trading environment."""

import pytest
import numpy as np

from src.environment.trading_env import TradingEnvironment
from src.environment.position_manager import PositionManager, PositionSide
from src.environment.reward_calculator import RewardCalculator
from src.config.constants import Action


class TestPositionManager:
    """Test position management."""
    
    def test_open_long_position(self):
        manager = PositionManager(lot_size=0.3, stop_loss_usd=300, take_profit_usd=500)
        
        position = manager.open_position(PositionSide.LONG, 30.0)
        
        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.entry_price == 30.0
        assert position.volume == 0.3
        assert manager.has_position()
    
    def test_open_short_position(self):
        manager = PositionManager(lot_size=0.3)
        
        position = manager.open_position(PositionSide.SHORT, 30.0)
        
        assert position is not None
        assert position.side == PositionSide.SHORT
        assert position.is_short
    
    def test_cannot_open_second_position(self):
        manager = PositionManager()
        
        manager.open_position(PositionSide.LONG, 30.0)
        second = manager.open_position(PositionSide.SHORT, 30.0)
        
        assert second is None
        assert manager.get_position().side == PositionSide.LONG
    
    def test_close_position(self):
        manager = PositionManager()
        
        manager.open_position(PositionSide.LONG, 30.0)
        manager.update_price(30.5)
        
        closed = manager.close_position(30.5)
        
        assert closed is not None
        assert closed.realized_pnl > 0
        assert not manager.has_position()
    
    def test_stop_loss_trigger(self):
        manager = PositionManager(lot_size=0.3, stop_loss_usd=300)
        
        manager.open_position(PositionSide.LONG, 30.0)
        
        # Move price down significantly
        should_close, reason = manager.update_price(29.5)
        
        # With 0.3 lots and 5000 oz/lot, a $0.5 move = $750 loss
        assert should_close
        assert reason.value == "stop_loss"
    
    def test_take_profit_trigger(self):
        manager = PositionManager(lot_size=0.3, take_profit_usd=500)
        
        manager.open_position(PositionSide.LONG, 30.0)
        
        # Move price up significantly
        should_close, reason = manager.update_price(30.5)
        
        assert should_close
        assert reason.value == "take_profit"


class TestRewardCalculator:
    """Test reward calculation."""
    
    def test_mape_calculation(self):
        calculator = RewardCalculator()
        
        predicted = np.array([30.0, 30.1, 29.9, 30.05, 1000])
        actual = np.array([30.0, 30.0, 30.0, 30.0, 1000])
        
        mape = calculator.calculate_mape(predicted, actual)
        
        assert mape > 0
        assert mape < 10  # Should be small percentage
    
    def test_reward_with_loss_weighting(self):
        calculator = RewardCalculator()
        
        predicted = np.array([30.0, 30.1, 29.9, 30.05, 1000])
        actual = np.array([30.0, 30.0, 30.0, 30.0, 1000])
        
        # No loss
        result1 = calculator.calculate_reward(predicted, actual, total_loss=0)
        
        # With loss
        calculator.reset()
        result2 = calculator.calculate_reward(predicted, actual, total_loss=500)
        
        # Higher loss should result in larger penalty (more negative reward)
        assert result2.prediction_penalty > result1.prediction_penalty
    
    def test_pnl_reward(self):
        calculator = RewardCalculator()
        
        predicted = np.array([30.0, 30.0, 30.0, 30.0, 1000])
        actual = np.array([30.0, 30.0, 30.0, 30.0, 1000])
        
        result = calculator.calculate_reward(
            predicted, actual,
            total_loss=0,
            realized_pnl=100,
        )
        
        assert result.pnl_reward > 0


class TestTradingEnvironment:
    """Test the trading environment."""
    
    @pytest.fixture
    def env(self):
        return TradingEnvironment(
            sequence_length=10,
            lot_size=0.3,
            stop_loss_usd=300,
            take_profit_usd=500,
            max_loss_usd=1000,
        )
    
    @pytest.fixture
    def sample_candle(self):
        return np.array([30.0, 30.1, 29.9, 30.05, 1000], dtype=np.float32)
    
    def test_reset(self, env, sample_candle):
        # Create initial candles
        initial_candles = [sample_candle for _ in range(10)]
        
        obs, info = env.reset(options={"initial_candles": initial_candles})
        
        assert "candles" in obs
        assert "position" in obs
        assert "account" in obs
        assert obs["candles"].shape == (10, 5)
    
    def test_step(self, env, sample_candle):
        initial_candles = [sample_candle for _ in range(10)]
        obs, _ = env.reset(options={"initial_candles": initial_candles})
        
        # Add new candle
        env.receive_candle(sample_candle)
        
        action = {
            "prediction": sample_candle,
            "trading_action": Action.NONE.value,
        }
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert not terminated
        assert "step" in info
    
    def test_buy_action(self, env, sample_candle):
        initial_candles = [sample_candle for _ in range(10)]
        obs, _ = env.reset(options={"initial_candles": initial_candles})
        
        env.receive_candle(sample_candle)
        
        action = {
            "prediction": sample_candle,
            "trading_action": Action.BUY.value,
        }
        
        _, _, _, _, info = env.step(action)
        
        assert info["has_position"]
        assert info["trading_action"] == "BUY"
    
    def test_episode_termination_on_max_loss(self, env, sample_candle):
        initial_candles = [sample_candle for _ in range(10)]
        obs, _ = env.reset(options={"initial_candles": initial_candles})
        
        # Open position
        env.receive_candle(sample_candle)
        action = {
            "prediction": sample_candle,
            "trading_action": Action.BUY.value,
        }
        env.step(action)
        
        # Move price down significantly to trigger max loss
        bad_candle = np.array([28.0, 28.1, 27.5, 27.5, 1000], dtype=np.float32)
        env.receive_candle(bad_candle)
        
        action = {
            "prediction": bad_candle,
            "trading_action": Action.NONE.value,
        }
        
        _, _, terminated, _, _ = env.step(action)
        
        # Should terminate due to exceeding max loss
        assert terminated

