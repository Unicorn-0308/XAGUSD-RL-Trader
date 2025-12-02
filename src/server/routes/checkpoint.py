"""Checkpoint management API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.config.settings import get_settings
from src.server.dependencies import get_agent_manager
from src.server.schemas import (
    CheckpointListResponse,
    CheckpointInfo,
    CheckpointLoadRequest,
    CheckpointSaveRequest,
    CheckpointActionResponse,
)
from src.utils.checkpoint import CheckpointManager


router = APIRouter()


def get_checkpoint_manager() -> CheckpointManager:
    """Get checkpoint manager instance."""
    settings = get_settings()
    return CheckpointManager(settings.checkpoint_dir)


@router.get("/list", response_model=CheckpointListResponse)
async def list_checkpoints() -> CheckpointListResponse:
    """List all available checkpoints.
    
    Returns list of checkpoints with metadata.
    """
    manager = get_checkpoint_manager()
    checkpoints = manager.list_checkpoints()
    
    best = manager.get_best_checkpoint()
    latest = manager.get_latest_checkpoint()
    
    return CheckpointListResponse(
        checkpoints=[
            CheckpointInfo(
                path=cp["path"],
                filename=cp["filename"],
                epoch=cp.get("epoch"),
                timestamp=cp.get("timestamp"),
                metadata=cp.get("metadata", {}),
            )
            for cp in checkpoints
        ],
        best_checkpoint=str(best) if best else None,
        latest_checkpoint=str(latest) if latest else None,
    )


@router.post("/load", response_model=CheckpointActionResponse)
async def load_checkpoint(request: CheckpointLoadRequest) -> CheckpointActionResponse:
    """Load a specific checkpoint.
    
    Loads model weights (and optionally optimizer state) from a checkpoint.
    """
    agent_manager = get_agent_manager()
    checkpoint_manager = get_checkpoint_manager()
    
    # Initialize agent if not done
    agent_manager.initialize()
    
    checkpoint_path = Path(request.path)
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {checkpoint_path}",
        )
    
    try:
        optimizer = agent_manager.agent.optimizer if request.load_optimizer else None
        metadata = checkpoint_manager.load(
            agent_manager.model,
            optimizer,
            checkpoint_path=checkpoint_path,
        )
        
        return CheckpointActionResponse(
            success=True,
            message="Checkpoint loaded successfully",
            path=str(checkpoint_path),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}",
        )


@router.post("/save", response_model=CheckpointActionResponse)
async def save_checkpoint(request: CheckpointSaveRequest) -> CheckpointActionResponse:
    """Save current model state.
    
    Saves the current model weights and optimizer state to a checkpoint.
    """
    agent_manager = get_agent_manager()
    checkpoint_manager = get_checkpoint_manager()
    
    if agent_manager.model is None:
        raise HTTPException(
            status_code=400,
            detail="Agent not initialized",
        )
    
    try:
        metadata = {
            "agent_state": agent_manager.get_state(),
            "manual_save": True,
        }
        
        if request.name:
            metadata["name"] = request.name
        
        path = checkpoint_manager.save(
            agent_manager.model,
            agent_manager.agent.optimizer,
            metadata,
            epoch=agent_manager.state.total_steps,
        )
        
        return CheckpointActionResponse(
            success=True,
            message="Checkpoint saved successfully",
            path=str(path),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save checkpoint: {str(e)}",
        )


@router.post("/load-best", response_model=CheckpointActionResponse)
async def load_best_checkpoint() -> CheckpointActionResponse:
    """Load the best performing checkpoint.
    
    Loads the checkpoint with the best reward achieved during training.
    """
    agent_manager = get_agent_manager()
    checkpoint_manager = get_checkpoint_manager()
    
    best = checkpoint_manager.get_best_checkpoint()
    if best is None:
        raise HTTPException(
            status_code=404,
            detail="No best checkpoint found",
        )
    
    agent_manager.initialize()
    
    try:
        metadata = checkpoint_manager.load(
            agent_manager.model,
            agent_manager.agent.optimizer,
            load_best=True,
        )
        
        return CheckpointActionResponse(
            success=True,
            message="Best checkpoint loaded successfully",
            path=str(best),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}",
        )


@router.post("/load-latest", response_model=CheckpointActionResponse)
async def load_latest_checkpoint() -> CheckpointActionResponse:
    """Load the latest checkpoint.
    
    Loads the most recently saved checkpoint.
    """
    agent_manager = get_agent_manager()
    checkpoint_manager = get_checkpoint_manager()
    
    latest = checkpoint_manager.get_latest_checkpoint()
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail="No checkpoint found",
        )
    
    agent_manager.initialize()
    
    try:
        metadata = checkpoint_manager.load(
            agent_manager.model,
            agent_manager.agent.optimizer,
        )
        
        return CheckpointActionResponse(
            success=True,
            message="Latest checkpoint loaded successfully",
            path=str(latest),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}",
        )

