"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Agent Schemas
# =============================================================================

class AgentStartRequest(BaseModel):
    """Request to start the agent."""
    
    load_checkpoint: bool = Field(
        default=True,
        description="Whether to load the latest checkpoint",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Specific checkpoint path to load",
    )


class AgentStatusResponse(BaseModel):
    """Response with agent status."""
    
    status: str = Field(description="Current agent status")
    started_at: datetime | None = Field(description="When agent was started")
    total_steps: int = Field(description="Total steps executed")
    total_trades: int = Field(description="Total trades executed")
    total_pnl: float = Field(description="Total profit/loss")
    current_position: dict | None = Field(description="Current open position")
    last_prediction: list[float] | None = Field(description="Last candle prediction")
    last_action: str | None = Field(description="Last trading action")
    error_message: str | None = Field(description="Error message if any")


class AgentActionResponse(BaseModel):
    """Response from agent action."""
    
    success: bool
    message: str


# =============================================================================
# Training Schemas
# =============================================================================

class PretrainRequest(BaseModel):
    """Request to start pre-training."""
    
    csv_path: str = Field(description="Path to CSV data file")
    total_timesteps: int = Field(
        default=1_000_000,
        description="Total training timesteps",
    )
    resume_from: str | None = Field(
        default=None,
        description="Checkpoint to resume from",
    )


class TrainingStatusResponse(BaseModel):
    """Response with training status."""
    
    status: str = Field(description="Current training status")
    progress: float = Field(description="Training progress (0-1)")
    timesteps: int = Field(description="Current timesteps")
    episodes: int = Field(description="Completed episodes")
    best_reward: float = Field(description="Best episode reward")
    started_at: datetime | None = Field(description="When training started")
    error_message: str | None = Field(description="Error message if any")


class TrainingActionResponse(BaseModel):
    """Response from training action."""
    
    success: bool
    message: str


# =============================================================================
# Data Schemas
# =============================================================================

class CandleResponse(BaseModel):
    """Single candle data."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesRequest(BaseModel):
    """Request for candle data."""
    
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class CandlesResponse(BaseModel):
    """Response with candle data."""
    
    candles: list[CandleResponse]
    total: int


class TradeRecord(BaseModel):
    """Single trade record."""
    
    position_id: str
    side: str
    entry_price: float
    exit_price: float | None
    volume: float
    open_time: datetime
    close_time: datetime | None
    realized_pnl: float | None
    close_reason: str | None


class TradesRequest(BaseModel):
    """Request for trade history."""
    
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class TradesResponse(BaseModel):
    """Response with trade history."""
    
    trades: list[TradeRecord]
    total: int
    summary: dict


class MetricsResponse(BaseModel):
    """Response with performance metrics."""
    
    total_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float | None
    max_drawdown: float | None
    prediction_accuracy: float | None


# =============================================================================
# Checkpoint Schemas
# =============================================================================

class CheckpointInfo(BaseModel):
    """Checkpoint information."""
    
    path: str
    filename: str
    epoch: int | None
    timestamp: str | None
    metadata: dict


class CheckpointListResponse(BaseModel):
    """Response with checkpoint list."""
    
    checkpoints: list[CheckpointInfo]
    best_checkpoint: str | None
    latest_checkpoint: str | None


class CheckpointLoadRequest(BaseModel):
    """Request to load a checkpoint."""
    
    path: str = Field(description="Path to checkpoint file")
    load_optimizer: bool = Field(
        default=True,
        description="Whether to load optimizer state",
    )


class CheckpointSaveRequest(BaseModel):
    """Request to save a checkpoint."""
    
    name: str | None = Field(
        default=None,
        description="Optional name for the checkpoint",
    )


class CheckpointActionResponse(BaseModel):
    """Response from checkpoint action."""
    
    success: bool
    message: str
    path: str | None = None


# =============================================================================
# WebSocket Schemas
# =============================================================================

class WSMessage(BaseModel):
    """WebSocket message."""
    
    type: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WSSubscribeMessage(BaseModel):
    """WebSocket subscribe message."""
    
    type: str = "subscribe"
    channel: str


class WSUnsubscribeMessage(BaseModel):
    """WebSocket unsubscribe message."""
    
    type: str = "unsubscribe"
    channel: str

