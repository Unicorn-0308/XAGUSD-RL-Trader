"""Data API routes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.config.settings import get_settings
from src.server.dependencies import get_agent_manager
from src.server.schemas import (
    CandleResponse,
    CandlesResponse,
    TradesResponse,
    TradeRecord,
    MetricsResponse,
)


router = APIRouter()


@router.get("/candles", response_model=CandlesResponse)
async def get_candles(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> CandlesResponse:
    """Get historical candle data.
    
    Returns recent candles from the agent's buffer.
    """
    manager = get_agent_manager()
    
    if manager.env is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized",
        )
    
    buffer = manager.env.candle_buffer
    all_candles = list(buffer)
    
    total = len(all_candles)
    candles = all_candles[offset:offset + limit]
    
    return CandlesResponse(
        candles=[
            CandleResponse(
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
            for c in candles
        ],
        total=total,
    )


@router.get("/trades", response_model=TradesResponse)
async def get_trades(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> TradesResponse:
    """Get trade history.
    
    Returns historical trades from the position manager.
    """
    manager = get_agent_manager()
    
    if manager.env is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized",
        )
    
    position_manager = manager.env.position_manager
    history = position_manager.get_history()
    stats = position_manager.get_statistics()
    
    total = len(history)
    trades = history[offset:offset + limit]
    
    return TradesResponse(
        trades=[
            TradeRecord(
                position_id=t.position_id,
                side=t.side.value,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                volume=t.volume,
                open_time=t.open_time,
                close_time=t.close_time,
                realized_pnl=t.realized_pnl,
                close_reason=t.close_reason.value,
            )
            for t in trades
        ],
        total=total,
        summary=stats,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get performance metrics.
    
    Returns aggregated trading performance metrics.
    """
    manager = get_agent_manager()
    
    if manager.env is None:
        # Return empty metrics
        return MetricsResponse(
            total_pnl=0.0,
            win_rate=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=None,
            max_drawdown=None,
            prediction_accuracy=None,
        )
    
    stats = manager.env.position_manager.get_statistics()
    
    return MetricsResponse(
        total_pnl=stats.get("total_pnl", 0.0),
        win_rate=stats.get("win_rate", 0.0),
        total_trades=stats.get("total_trades", 0),
        winning_trades=stats.get("winning_trades", 0),
        losing_trades=stats.get("losing_trades", 0),
        avg_win=stats.get("avg_win", 0.0),
        avg_loss=stats.get("avg_loss", 0.0),
        profit_factor=stats.get("profit_factor", 0.0),
        sharpe_ratio=None,  # TODO: Calculate
        max_drawdown=None,  # TODO: Calculate
        prediction_accuracy=None,  # TODO: Calculate
    )


@router.get("/csv-files")
async def list_csv_files() -> dict[str, Any]:
    """List available CSV files for training.
    
    Returns list of CSV files in the historical data directory.
    """
    settings = get_settings()
    data_dir = settings.historical_data_dir
    
    if not data_dir.exists():
        return {"files": [], "directory": str(data_dir)}
    
    files = []
    for file_path in data_dir.glob("*.csv"):
        stat = file_path.stat()
        files.append({
            "name": file_path.name,
            "path": str(file_path),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    
    return {
        "files": sorted(files, key=lambda x: x["name"]),
        "directory": str(data_dir),
    }

