"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import Settings, get_settings
from src.server.routes import (
    agent_router,
    training_router,
    data_router,
    checkpoint_router,
)
from src.server.websocket_manager import WebSocketManager
from src.server.dependencies import (
    get_agent_manager,
    get_training_manager,
    cleanup_managers,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    settings = get_settings()
    logger.info(
        "Starting XAGUSD RL Trader API",
        environment=settings.app_env,
        host=settings.server_host,
        port=settings.server_port,
    )
    
    # Initialize managers
    app.state.websocket_manager = WebSocketManager()
    app.state.agent_manager = get_agent_manager()
    app.state.training_manager = get_training_manager()
    
    # Ensure directories exist
    settings.ensure_directories()
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    await cleanup_managers()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()
    
    app = FastAPI(
        title="XAGUSD RL Trader API",
        description="Deep RL Trading System API for XAGUSD",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
    app.include_router(training_router, prefix="/api/training", tags=["Training"])
    app.include_router(data_router, prefix="/api/data", tags=["Data"])
    app.include_router(checkpoint_router, prefix="/api/checkpoint", tags=["Checkpoint"])
    
    # WebSocket route
    from src.server.websocket_manager import websocket_endpoint
    app.add_api_websocket_route("/ws/live", websocket_endpoint)
    
    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
        }
    
    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """Root endpoint."""
        return {
            "message": "XAGUSD RL Trader API",
            "docs": "/docs",
        }
    
    return app


# For uvicorn direct run
app = create_app()

