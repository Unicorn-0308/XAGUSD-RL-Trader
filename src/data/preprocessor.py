"""Data preprocessing and normalization for trading data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class NormalizationStats:
    """Statistics for normalization."""
    
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min_val": self.min_val.tolist(),
            "max_val": self.max_val.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> NormalizationStats:
        """Create from dictionary."""
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            min_val=np.array(data["min_val"], dtype=np.float32),
            max_val=np.array(data["max_val"], dtype=np.float32),
        )


@dataclass
class Preprocessor:
    """Preprocessor for normalizing OHLCV candle data.
    
    Supports multiple normalization strategies:
    - zscore: (x - mean) / std
    - minmax: (x - min) / (max - min)
    - returns: percentage returns from previous value
    - log_returns: log returns from previous value
    - rolling_zscore: z-score using rolling window statistics
    
    The preprocessor can be fit on historical data and then applied to new data
    consistently, which is important for model inference.
    """
    
    method: Literal["zscore", "minmax", "returns", "log_returns", "rolling_zscore"] = "rolling_zscore"
    rolling_window: int = 100
    epsilon: float = 1e-8
    clip_value: float = 10.0
    
    # Learned statistics
    _stats: NormalizationStats | None = field(default=None, init=False)
    _is_fitted: bool = field(default=False, init=False)
    
    # Rolling state for online processing
    _rolling_buffer: np.ndarray | None = field(default=None, init=False)

    def fit(self, data: np.ndarray) -> Preprocessor:
        """Fit the preprocessor on training data.
        
        Args:
            data: Array of shape [N, 5] with OHLCV data
            
        Returns:
            self for chaining
        """
        if data.ndim != 2 or data.shape[1] != 5:
            raise ValueError(f"Expected shape [N, 5], got {data.shape}")
        
        self._stats = NormalizationStats(
            mean=np.mean(data, axis=0).astype(np.float32),
            std=np.std(data, axis=0).astype(np.float32) + self.epsilon,
            min_val=np.min(data, axis=0).astype(np.float32),
            max_val=np.max(data, axis=0).astype(np.float32),
        )
        self._is_fitted = True
        
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted normalization.
        
        Args:
            data: Array of shape [N, 5] or [B, N, 5] with OHLCV data
            
        Returns:
            Normalized data with same shape
        """
        original_shape = data.shape
        
        # Handle batch dimension
        if data.ndim == 3:
            batch_size, seq_len, features = data.shape
            data = data.reshape(-1, features)
            reshaped = True
        else:
            reshaped = False
        
        if self.method == "zscore":
            result = self._zscore_transform(data)
        elif self.method == "minmax":
            result = self._minmax_transform(data)
        elif self.method == "returns":
            result = self._returns_transform(data)
        elif self.method == "log_returns":
            result = self._log_returns_transform(data)
        elif self.method == "rolling_zscore":
            result = self._rolling_zscore_transform(data)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Clip extreme values
        result = np.clip(result, -self.clip_value, self.clip_value)
        
        if reshaped:
            result = result.reshape(original_shape)
        
        return result.astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization to get original scale values.
        
        Note: This is approximate for returns-based methods.
        
        Args:
            data: Normalized data of shape [N, 5] or [B, N, 5]
            
        Returns:
            Data in original scale
        """
        if not self._is_fitted or self._stats is None:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        original_shape = data.shape
        if data.ndim == 3:
            batch_size, seq_len, features = data.shape
            data = data.reshape(-1, features)
            reshaped = True
        else:
            reshaped = False
        
        if self.method == "zscore":
            result = data * self._stats.std + self._stats.mean
        elif self.method == "minmax":
            range_val = self._stats.max_val - self._stats.min_val + self.epsilon
            result = data * range_val + self._stats.min_val
        else:
            # For returns-based methods, inverse is not straightforward
            # Return z-score inverse as approximation
            result = data * self._stats.std + self._stats.mean
        
        if reshaped:
            result = result.reshape(original_shape)
        
        return result.astype(np.float32)

    def _zscore_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        if not self._is_fitted or self._stats is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return (data - self._stats.mean) / self._stats.std

    def _minmax_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to [0, 1] range."""
        if not self._is_fitted or self._stats is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        range_val = self._stats.max_val - self._stats.min_val + self.epsilon
        return (data - self._stats.min_val) / range_val

    def _returns_transform(self, data: np.ndarray) -> np.ndarray:
        """Calculate percentage returns."""
        if len(data) < 2:
            return np.zeros_like(data)
        
        # First row is zeros (no previous value)
        returns = np.zeros_like(data)
        returns[1:] = (data[1:] - data[:-1]) / (data[:-1] + self.epsilon) * 100
        return returns

    def _log_returns_transform(self, data: np.ndarray) -> np.ndarray:
        """Calculate log returns."""
        if len(data) < 2:
            return np.zeros_like(data)
        
        # Ensure positive values
        data = np.maximum(data, self.epsilon)
        
        returns = np.zeros_like(data)
        returns[1:] = np.log(data[1:] / data[:-1]) * 100
        return returns

    def _rolling_zscore_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling z-score normalization.
        
        This is more suitable for non-stationary financial data as it
        adapts to local statistics rather than global ones.
        """
        result = np.zeros_like(data)
        
        for i in range(len(data)):
            start_idx = max(0, i - self.rolling_window + 1)
            window = data[start_idx:i + 1]
            
            if len(window) < 2:
                # Not enough data, use global stats if available
                if self._is_fitted and self._stats is not None:
                    result[i] = (data[i] - self._stats.mean) / self._stats.std
                else:
                    result[i] = 0
            else:
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0) + self.epsilon
                result[i] = (data[i] - mean) / std
        
        return result

    def normalize_single(self, candle_array: np.ndarray) -> np.ndarray:
        """Normalize a single candle using rolling buffer.
        
        This method maintains state for online processing.
        
        Args:
            candle_array: Array of shape [5] with OHLCV data
            
        Returns:
            Normalized array of shape [5]
        """
        if candle_array.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {candle_array.shape}")
        
        # Initialize or update rolling buffer
        if self._rolling_buffer is None:
            self._rolling_buffer = candle_array.reshape(1, -1)
        else:
            self._rolling_buffer = np.vstack([self._rolling_buffer, candle_array])
            # Keep only rolling_window elements
            if len(self._rolling_buffer) > self.rolling_window:
                self._rolling_buffer = self._rolling_buffer[-self.rolling_window:]
        
        # Calculate rolling statistics
        if len(self._rolling_buffer) < 2:
            if self._is_fitted and self._stats is not None:
                return ((candle_array - self._stats.mean) / self._stats.std).astype(np.float32)
            return np.zeros(5, dtype=np.float32)
        
        mean = np.mean(self._rolling_buffer, axis=0)
        std = np.std(self._rolling_buffer, axis=0) + self.epsilon
        
        result = (candle_array - mean) / std
        result = np.clip(result, -self.clip_value, self.clip_value)
        
        return result.astype(np.float32)

    def reset_rolling_state(self) -> None:
        """Reset the rolling buffer state."""
        self._rolling_buffer = None

    def get_stats(self) -> NormalizationStats | None:
        """Get the fitted statistics."""
        return self._stats

    def set_stats(self, stats: NormalizationStats) -> None:
        """Set statistics from saved values."""
        self._stats = stats
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor is fitted."""
        return self._is_fitted


def create_features(
    candles: np.ndarray,
    include_returns: bool = True,
    include_volatility: bool = True,
    include_momentum: bool = True,
) -> np.ndarray:
    """Create additional features from OHLCV data.
    
    Args:
        candles: Array of shape [N, 5] with OHLCV data
        include_returns: Include return-based features
        include_volatility: Include volatility features
        include_momentum: Include momentum features
        
    Returns:
        Extended feature array
    """
    features = [candles.copy()]
    
    if include_returns:
        # Close-to-close returns
        close = candles[:, 3]
        returns = np.zeros_like(close)
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
        features.append(returns.reshape(-1, 1))
        
        # High-low range normalized by close
        hl_range = (candles[:, 1] - candles[:, 2]) / (candles[:, 3] + 1e-8)
        features.append(hl_range.reshape(-1, 1))
        
        # Body size (close - open) normalized
        body = (candles[:, 3] - candles[:, 0]) / (candles[:, 3] + 1e-8)
        features.append(body.reshape(-1, 1))
    
    if include_volatility:
        # Rolling volatility (standard deviation of returns)
        close = candles[:, 3]
        volatility = np.zeros(len(close))
        window = 20
        for i in range(window, len(close)):
            window_returns = np.diff(close[i-window:i]) / (close[i-window:i-1] + 1e-8)
            volatility[i] = np.std(window_returns)
        features.append(volatility.reshape(-1, 1))
    
    if include_momentum:
        # RSI-like momentum indicator
        close = candles[:, 3]
        momentum = np.zeros(len(close))
        window = 14
        for i in range(window, len(close)):
            changes = np.diff(close[i-window:i+1])
            gains = np.maximum(changes, 0)
            losses = np.maximum(-changes, 0)
            avg_gain = np.mean(gains) + 1e-8
            avg_loss = np.mean(losses) + 1e-8
            rs = avg_gain / avg_loss
            momentum[i] = (100 - (100 / (1 + rs))) / 100 - 0.5  # Normalized to [-0.5, 0.5]
        features.append(momentum.reshape(-1, 1))
    
    return np.hstack(features).astype(np.float32)

