"""Checkpoint management for saving and loading model state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.config.constants import (
    CHECKPOINT_FILENAME_FORMAT,
    BEST_MODEL_FILENAME,
    LATEST_MODEL_FILENAME,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoints for training.
    
    Features:
    - Save periodic checkpoints
    - Save best model based on metric
    - Save latest model for resume
    - Keep N most recent checkpoints
    - Load from checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_checkpoints: int = 5,
    ) -> None:
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of periodic checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        self._checkpoints: list[Path] = []
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self) -> None:
        """Load list of existing checkpoints."""
        pattern = "checkpoint_epoch*.pt"
        self._checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metadata: dict[str, Any],
        epoch: int,
        is_final: bool = False,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metadata: Additional metadata (training state, etc.)
            epoch: Current epoch/rollout number
            is_final: Whether this is the final checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_final:
            filename = f"checkpoint_final_{timestamp}.pt"
        else:
            filename = CHECKPOINT_FILENAME_FORMAT.format(epoch=epoch, timestamp=timestamp)
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "timestamp": timestamp,
            "metadata": metadata,
        }
        
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved", path=str(filepath), epoch=epoch)
        
        # Also save as latest
        self._save_latest(checkpoint)
        
        # Manage checkpoint history
        if not is_final:
            self._checkpoints.append(filepath)
            self._cleanup_old_checkpoints()
        
        return filepath
    
    def save_best(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metadata: dict[str, Any],
    ) -> Path:
        """Save the best model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        filepath = self.checkpoint_dir / BEST_MODEL_FILENAME
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }
        
        torch.save(checkpoint, filepath)
        logger.info("Best model saved", path=str(filepath))
        
        return filepath
    
    def _save_latest(self, checkpoint: dict[str, Any]) -> None:
        """Save as latest checkpoint for easy resume."""
        filepath = self.checkpoint_dir / LATEST_MODEL_FILENAME
        torch.save(checkpoint, filepath)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max."""
        while len(self._checkpoints) > self.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug("Removed old checkpoint", path=str(oldest))
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        checkpoint_path: Path | str | None = None,
        load_best: bool = False,
    ) -> dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Specific checkpoint path (if None, loads latest)
            load_best: Whether to load best model instead of latest
            
        Returns:
            Checkpoint metadata
        """
        if checkpoint_path:
            filepath = Path(checkpoint_path)
        elif load_best:
            filepath = self.checkpoint_dir / BEST_MODEL_FILENAME
        else:
            filepath = self.checkpoint_dir / LATEST_MODEL_FILENAME
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info("Checkpoint loaded", path=str(filepath))
        
        return checkpoint.get("metadata", {})
    
    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint if exists."""
        latest = self.checkpoint_dir / LATEST_MODEL_FILENAME
        return latest if latest.exists() else None
    
    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint if exists."""
        best = self.checkpoint_dir / BEST_MODEL_FILENAME
        return best if best.exists() else None
    
    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint info dicts
        """
        checkpoints = []
        
        for filepath in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(filepath, map_location="cpu")
                checkpoints.append({
                    "path": str(filepath),
                    "filename": filepath.name,
                    "epoch": checkpoint.get("epoch"),
                    "timestamp": checkpoint.get("timestamp"),
                    "metadata": checkpoint.get("metadata", {}),
                })
            except Exception as e:
                logger.warning("Failed to load checkpoint info", path=str(filepath), error=str(e))
        
        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""), reverse=True)


def save_model_only(model: nn.Module, path: Path | str) -> None:
    """Save only model weights (for deployment).
    
    Args:
        model: Model to save
        path: Path to save to
    """
    torch.save(model.state_dict(), path)


def load_model_only(model: nn.Module, path: Path | str, device: str = "cpu") -> None:
    """Load only model weights.
    
    Args:
        model: Model to load into
        path: Path to load from
        device: Device to map to
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

