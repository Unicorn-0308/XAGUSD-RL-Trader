"""WebSocket manager for real-time updates."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.utils.logging import get_logger


logger = get_logger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates.
    
    Features:
    - Multiple concurrent connections
    - Broadcast to all clients
    - Channel-based subscriptions
    - Automatic reconnection handling
    """
    
    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        self._connections: dict[str, WebSocket] = {}
        self._subscriptions: dict[str, set[str]] = {}  # channel -> client_ids
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str | None = None,
    ) -> str:
        """Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
            
        Returns:
            Assigned client ID
        """
        await websocket.accept()
        
        if client_id is None:
            client_id = f"client_{datetime.now().timestamp()}"
        
        async with self._lock:
            self._connections[client_id] = websocket
        
        logger.info("WebSocket connected", client_id=client_id)
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        return client_id
    
    async def disconnect(self, client_id: str) -> None:
        """Handle WebSocket disconnection.
        
        Args:
            client_id: Client identifier
        """
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
            
            # Remove from all subscriptions
            for channel in list(self._subscriptions.keys()):
                self._subscriptions[channel].discard(client_id)
        
        logger.info("WebSocket disconnected", client_id=client_id)
    
    async def subscribe(self, client_id: str, channel: str) -> None:
        """Subscribe a client to a channel.
        
        Args:
            client_id: Client identifier
            channel: Channel name
        """
        async with self._lock:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = set()
            self._subscriptions[channel].add(client_id)
        
        logger.debug("Client subscribed", client_id=client_id, channel=channel)
    
    async def unsubscribe(self, client_id: str, channel: str) -> None:
        """Unsubscribe a client from a channel.
        
        Args:
            client_id: Client identifier
            channel: Channel name
        """
        async with self._lock:
            if channel in self._subscriptions:
                self._subscriptions[channel].discard(client_id)
    
    async def send_to_client(self, client_id: str, data: dict[str, Any]) -> bool:
        """Send data to a specific client.
        
        Args:
            client_id: Client identifier
            data: Data to send
            
        Returns:
            True if sent successfully
        """
        if client_id not in self._connections:
            return False
        
        websocket = self._connections[client_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
                return True
        except Exception as e:
            logger.error("Failed to send to client", client_id=client_id, error=str(e))
            await self.disconnect(client_id)
        
        return False
    
    async def broadcast(self, data: dict[str, Any]) -> int:
        """Broadcast data to all connected clients.
        
        Args:
            data: Data to broadcast
            
        Returns:
            Number of clients reached
        """
        sent_count = 0
        
        # Copy to avoid modification during iteration
        client_ids = list(self._connections.keys())
        
        for client_id in client_ids:
            if await self.send_to_client(client_id, data):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_channel(self, channel: str, data: dict[str, Any]) -> int:
        """Broadcast data to all clients subscribed to a channel.
        
        Args:
            channel: Channel name
            data: Data to broadcast
            
        Returns:
            Number of clients reached
        """
        if channel not in self._subscriptions:
            return 0
        
        sent_count = 0
        client_ids = list(self._subscriptions[channel])
        
        for client_id in client_ids:
            if await self.send_to_client(client_id, data):
                sent_count += 1
        
        return sent_count
    
    @property
    def connection_count(self) -> int:
        """Get number of connected clients."""
        return len(self._connections)
    
    def get_connected_clients(self) -> list[str]:
        """Get list of connected client IDs."""
        return list(self._connections.keys())


# Global WebSocket manager instance
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """Get the WebSocket manager singleton."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint handler.
    
    Handles the WebSocket connection lifecycle and message processing.
    
    Message format:
    {
        "type": "subscribe" | "unsubscribe" | "ping",
        "channel": "trading" | "training" | "logs",  # for subscribe/unsubscribe
        ...
    }
    """
    manager = get_ws_manager()
    client_id = await manager.connect(websocket)
    
    # Get managers for callbacks
    from src.server.dependencies import get_agent_manager, get_training_manager
    agent_manager = get_agent_manager()
    training_manager = get_training_manager()
    
    # Setup callbacks to forward updates
    async def forward_update(data: dict) -> None:
        await manager.send_to_client(client_id, data)
    
    def sync_forward(data: dict) -> None:
        asyncio.create_task(forward_update(data))
    
    agent_manager.add_callback(sync_forward)
    training_manager.add_callback(sync_forward)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                channel = data.get("channel")
                if channel:
                    await manager.subscribe(client_id, channel)
                    await manager.send_to_client(client_id, {
                        "type": "subscribed",
                        "channel": channel,
                    })
            
            elif msg_type == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    await manager.unsubscribe(client_id, channel)
                    await manager.send_to_client(client_id, {
                        "type": "unsubscribed",
                        "channel": channel,
                    })
            
            elif msg_type == "ping":
                await manager.send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                })
            
            elif msg_type == "get_status":
                await manager.send_to_client(client_id, {
                    "type": "status",
                    "agent": agent_manager.get_state(),
                    "training": training_manager.get_status(),
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
    finally:
        # Cleanup callbacks
        agent_manager.remove_callback(sync_forward)
        training_manager.remove_callback(sync_forward)
        await manager.disconnect(client_id)

