"""Training control API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.config.settings import get_settings
from src.server.dependencies import get_training_manager
from src.server.schemas import (
    PretrainRequest,
    TrainingStatusResponse,
    TrainingActionResponse,
)


router = APIRouter()


@router.post("/pretrain", response_model=TrainingActionResponse)
async def start_pretraining(request: PretrainRequest) -> TrainingActionResponse:
    """Start pre-training on CSV data.
    
    Begins training the agent on historical data from a CSV file.
    """
    manager = get_training_manager()
    settings = get_settings()
    
    if manager.status.status in ["pretraining", "online_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training is already in progress",
        )
    
    # Validate CSV path
    csv_path = Path(request.csv_path)
    if not csv_path.is_absolute():
        csv_path = settings.historical_data_dir / csv_path
    
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"CSV file not found: {csv_path}",
        )
    
    success = await manager.start_pretraining(
        csv_path=str(csv_path),
        total_timesteps=request.total_timesteps,
    )
    
    if success:
        return TrainingActionResponse(
            success=True,
            message="Pre-training started",
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=manager.status.error_message or "Failed to start training",
        )


@router.post("/stop", response_model=TrainingActionResponse)
async def stop_training() -> TrainingActionResponse:
    """Stop current training.
    
    Gracefully stops the training process and saves a checkpoint.
    """
    manager = get_training_manager()
    
    if manager.status.status not in ["pretraining", "online_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training is not running",
        )
    
    success = await manager.stop_training()
    
    if success:
        return TrainingActionResponse(
            success=True,
            message="Training stopped",
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to stop training",
        )


@router.get("/progress", response_model=TrainingStatusResponse)
async def get_training_progress() -> TrainingStatusResponse:
    """Get current training progress.
    
    Returns detailed training metrics and status.
    """
    manager = get_training_manager()
    status = manager.get_status()
    
    return TrainingStatusResponse(**status)


@router.post("/pause", response_model=TrainingActionResponse)
async def pause_training() -> TrainingActionResponse:
    """Pause training.
    
    Temporarily pauses the training process.
    """
    manager = get_training_manager()
    
    if manager.status.status not in ["pretraining", "online_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training is not running",
        )
    
    if manager.trainer:
        manager.trainer.pause()
    
    manager.status.status = "paused"
    
    return TrainingActionResponse(
        success=True,
        message="Training paused",
    )


@router.post("/resume", response_model=TrainingActionResponse)
async def resume_training() -> TrainingActionResponse:
    """Resume training.
    
    Resumes a paused training process.
    """
    manager = get_training_manager()
    
    if manager.status.status != "paused":
        raise HTTPException(
            status_code=400,
            detail="Training is not paused",
        )
    
    if manager.trainer:
        manager.trainer.resume()
    
    manager.status.status = "pretraining"  # or online_training
    
    return TrainingActionResponse(
        success=True,
        message="Training resumed",
    )

