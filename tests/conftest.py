"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_candles() -> np.ndarray:
    """Generate sample OHLCV candle data for testing."""
    np.random.seed(42)
    n_candles = 150
    
    # Generate realistic-looking price data
    base_price = 30.0  # Silver around $30
    returns = np.random.randn(n_candles) * 0.001  # 0.1% volatility
    
    closes = base_price * np.exp(np.cumsum(returns))
    
    candles = []
    for i, close in enumerate(closes):
        high = close * (1 + abs(np.random.randn()) * 0.001)
        low = close * (1 - abs(np.random.randn()) * 0.001)
        open_price = low + (high - low) * np.random.rand()
        volume = np.random.randint(100, 10000)
        
        candles.append([open_price, high, low, close, volume])
    
    return np.array(candles, dtype=np.float32)


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_candles: np.ndarray) -> Path:
    """Create a temporary CSV file with sample data."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(sample_candles))]
    
    # Create DataFrame
    df = pd.DataFrame(
        sample_candles,
        columns=["open", "high", "low", "close", "volume"]
    )
    df.insert(0, "timestamp", timestamps)
    
    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def sequence_length() -> int:
    """Default sequence length for testing."""
    return 120


@pytest.fixture
def hidden_size() -> int:
    """Default hidden size for testing."""
    return 256


@pytest.fixture
def batch_size() -> int:
    """Default batch size for testing."""
    return 8

