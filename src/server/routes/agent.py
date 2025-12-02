"""Agent control API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.server.dependencies import get_agent_manager
from src.server.schemas import (
    AgentStartRequest,
    AgentStatusResponse,
    AgentActionResponse,
)


router = APIRouter()


@router.post("/start", response_model=AgentActionResponse)
async def start_agent(request: AgentStartRequest) -> AgentActionResponse:
    """Start live trading agent.
    
    Initializes the agent and begins live trading with Match-Trader API.
    """
    manager = get_agent_manager()
    
    if manager.state.status == "running":
        raise HTTPException(
            status_code=400,
            detail="Agent is already running",
        )
    
    # Load checkpoint if requested
    if request.load_checkpoint:
        try:
            from src.utils.checkpoint import CheckpointManager
            from src.config.settings import get_settings
            
            settings = get_settings()
            checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
            
            manager.initialize()
            
            if request.checkpoint_path:
                checkpoint_mgr.load(
                    manager.model,
                    manager.agent.optimizer,
                    checkpoint_path=request.checkpoint_path,
                )
            elif checkpoint_mgr.get_latest_checkpoint():
                checkpoint_mgr.load(
                    manager.model,
                    manager.agent.optimizer,
                )
        except Exception as e:
            # Continue without checkpoint
            pass
    
    success = await manager.start()
    
    if success:
        return AgentActionResponse(
            success=True,
            message="Agent started successfully",
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=manager.state.error_message or "Failed to start agent",
        )


@router.post("/stop", response_model=AgentActionResponse)
async def stop_agent() -> AgentActionResponse:
    """Stop live trading agent.
    
    Gracefully stops trading and closes any open positions.
    """
    manager = get_agent_manager()
    
    if manager.state.status != "running":
        raise HTTPException(
            status_code=400,
            detail="Agent is not running",
        )
    
    success = await manager.stop()
    
    if success:
        return AgentActionResponse(
            success=True,
            message="Agent stopped successfully",
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to stop agent",
        )


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status() -> AgentStatusResponse:
    """Get current agent status.
    
    Returns the current state of the trading agent including
    position, PnL, and recent predictions.
    """
    manager = get_agent_manager()
    state = manager.get_state()
    
    return AgentStatusResponse(**state)


@router.post("/pause", response_model=AgentActionResponse)
async def pause_agent() -> AgentActionResponse:
    """Pause the trading agent.
    
    Temporarily pauses trading without closing positions.
    """
    manager = get_agent_manager()
    
    if manager.state.status != "running":
        raise HTTPException(
            status_code=400,
            detail="Agent is not running",
        )
    
    manager.state.status = "paused"
    
    return AgentActionResponse(
        success=True,
        message="Agent paused",
    )


@router.post("/resume", response_model=AgentActionResponse)
async def resume_agent() -> AgentActionResponse:
    """Resume the trading agent.
    
    Resumes trading after being paused.
    """
    manager = get_agent_manager()
    
    if manager.state.status != "paused":
        raise HTTPException(
            status_code=400,
            detail="Agent is not paused",
        )
    
    manager.state.status = "running"
    
    return AgentActionResponse(
        success=True,
        message="Agent resumed",
    )

