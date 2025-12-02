"""API routes module."""

from src.server.routes.agent import router as agent_router
from src.server.routes.training import router as training_router
from src.server.routes.data import router as data_router
from src.server.routes.checkpoint import router as checkpoint_router

__all__ = [
    "agent_router",
    "training_router",
    "data_router",
    "checkpoint_router",
]

