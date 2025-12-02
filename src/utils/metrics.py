"""Metrics logging for training using TensorBoard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore

from src.utils.logging import get_logger


logger = get_logger(__name__)


class MetricsWriter:
    """Write training metrics to TensorBoard.
    
    Provides a clean interface for logging scalars, histograms,
    and other metrics during training.
    """
    
    def __init__(
        self,
        log_dir: Path | str,
        experiment_name: str | None = None,
        flush_secs: int = 30,
    ) -> None:
        """Initialize metrics writer.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Optional experiment name for subdirectory
            flush_secs: How often to flush to disk
        """
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available. Metrics will not be logged.")
            self._writer = None
            return
        
        log_dir = Path(log_dir)
        
        if experiment_name:
            log_dir = log_dir / experiment_name
        else:
            # Use timestamp as experiment name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = log_dir / f"run_{timestamp}"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=flush_secs,
        )
        self.log_dir = log_dir
        
        logger.info("TensorBoard logging to", log_dir=str(log_dir))
    
    def add_scalar(
        self,
        tag: str,
        value: float | int,
        step: int,
    ) -> None:
        """Add a scalar value.
        
        Args:
            tag: Metric name (e.g., "train/loss")
            value: Scalar value
            step: Training step
        """
        if self._writer is None:
            return
        self._writer.add_scalar(tag, value, step)
    
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        step: int,
    ) -> None:
        """Add multiple scalars to the same plot.
        
        Args:
            main_tag: Main metric name
            tag_scalar_dict: Dict of {sub_tag: value}
            step: Training step
        """
        if self._writer is None:
            return
        self._writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def add_histogram(
        self,
        tag: str,
        values: np.ndarray | list,
        step: int,
        bins: str = "tensorflow",
    ) -> None:
        """Add a histogram.
        
        Args:
            tag: Metric name
            values: Array of values
            step: Training step
            bins: Binning method
        """
        if self._writer is None:
            return
        self._writer.add_histogram(tag, np.array(values), step, bins=bins)
    
    def add_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """Add text.
        
        Args:
            tag: Tag name
            text: Text content
            step: Training step
        """
        if self._writer is None:
            return
        self._writer.add_text(tag, text, step)
    
    def add_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float],
    ) -> None:
        """Add hyperparameters and their results.
        
        Args:
            hparam_dict: Hyperparameter dict
            metric_dict: Result metrics dict
        """
        if self._writer is None:
            return
        self._writer.add_hparams(hparam_dict, metric_dict)
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        step: int,
        additional_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length
            step: Training step
            additional_metrics: Additional metrics to log
        """
        self.add_scalar("episode/reward", reward, step)
        self.add_scalar("episode/length", length, step)
        self.add_scalar("episode/number", episode, step)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.add_scalar(f"episode/{key}", value, step)
    
    def log_training_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        additional_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log training step metrics.
        
        Args:
            step: Training step
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Entropy value
            additional_metrics: Additional metrics to log
        """
        self.add_scalar("train/policy_loss", policy_loss, step)
        self.add_scalar("train/value_loss", value_loss, step)
        self.add_scalar("train/entropy", entropy, step)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.add_scalar(f"train/{key}", value, step)
    
    def log_trading_metrics(
        self,
        step: int,
        total_pnl: float,
        win_rate: float,
        num_trades: int,
        prediction_accuracy: float | None = None,
    ) -> None:
        """Log trading-specific metrics.
        
        Args:
            step: Training step
            total_pnl: Total profit/loss
            win_rate: Win rate (0-1)
            num_trades: Number of trades
            prediction_accuracy: Optional prediction accuracy
        """
        self.add_scalar("trading/total_pnl", total_pnl, step)
        self.add_scalar("trading/win_rate", win_rate, step)
        self.add_scalar("trading/num_trades", num_trades, step)
        
        if prediction_accuracy is not None:
            self.add_scalar("trading/prediction_accuracy", prediction_accuracy, step)
    
    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self._writer is not None:
            self._writer.flush()
    
    def close(self) -> None:
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            logger.info("TensorBoard writer closed")


class MetricsAggregator:
    """Aggregate metrics over multiple steps for logging."""
    
    def __init__(self, window_size: int = 100) -> None:
        """Initialize aggregator.
        
        Args:
            window_size: Number of values to keep for averaging
        """
        self.window_size = window_size
        self._metrics: dict[str, list[float]] = {}
    
    def add(self, name: str, value: float) -> None:
        """Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(value)
        
        # Keep only window_size values
        if len(self._metrics[name]) > self.window_size:
            self._metrics[name] = self._metrics[name][-self.window_size:]
    
    def get_mean(self, name: str) -> float:
        """Get mean of metric values.
        
        Args:
            name: Metric name
            
        Returns:
            Mean value or 0 if no values
        """
        if name not in self._metrics or not self._metrics[name]:
            return 0.0
        return float(np.mean(self._metrics[name]))
    
    def get_std(self, name: str) -> float:
        """Get std of metric values.
        
        Args:
            name: Metric name
            
        Returns:
            Std value or 0 if no values
        """
        if name not in self._metrics or not self._metrics[name]:
            return 0.0
        return float(np.std(self._metrics[name]))
    
    def get_all_means(self) -> dict[str, float]:
        """Get means of all metrics.
        
        Returns:
            Dict of metric name to mean value
        """
        return {name: self.get_mean(name) for name in self._metrics}
    
    def clear(self, name: str | None = None) -> None:
        """Clear metric values.
        
        Args:
            name: Specific metric to clear, or None to clear all
        """
        if name is None:
            self._metrics.clear()
        elif name in self._metrics:
            self._metrics[name] = []

