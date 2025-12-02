"""FastAPI dependencies and state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import asyncio
from datetime import datetime

from src.config.settings import get_settings
from src.models.actor_critic import HybridActorCritic
from src.agent.ppo_agent import PPOAgent, PPOConfig
from src.agent.trainer import Trainer, TrainingConfig, TrainingState
from src.environment.trading_env import TradingEnvironment
from src.data.csv_loader import CSVDataLoader
from src.data.match_trader_client import MatchTraderClient, SimulatedMatchTraderClient
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class AgentState:
    """Current state of the trading agent."""
    
    status: str = "stopped"  # stopped, running, paused, error
    started_at: datetime | None = None
    total_steps: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    current_position: dict | None = None
    last_prediction: list[float] | None = None
    last_action: str | None = None
    error_message: str | None = None


@dataclass
class TrainingStatus:
    """Current training status."""
    
    status: str = "idle"  # idle, pretraining, online_training, paused, error
    progress: float = 0.0
    timesteps: int = 0
    episodes: int = 0
    best_reward: float = float("-inf")
    current_epoch: int = 0
    started_at: datetime | None = None
    error_message: str | None = None


class AgentManager:
    """Manages the trading agent lifecycle."""
    
    def __init__(self) -> None:
        """Initialize the agent manager."""
        self.settings = get_settings()
        
        self.model: HybridActorCritic | None = None
        self.agent: PPOAgent | None = None
        self.env: TradingEnvironment | None = None
        self.client: MatchTraderClient | None = None
        
        self.state = AgentState()
        self._running = False
        self._task: asyncio.Task | None = None
        self._callbacks: list[Callable[[dict], None]] = []
    
    def initialize(self) -> None:
        """Initialize model and agent."""
        if self.model is None:
            self.model = HybridActorCritic(
                sequence_length=self.settings.model_sequence_length,
                hidden_dim=self.settings.model_hidden_size,
                num_layers=self.settings.model_num_layers,
                num_heads=self.settings.model_attention_heads,
                dropout=self.settings.model_dropout,
                device=self.settings.device,
            )
            
            config = PPOConfig(
                learning_rate=self.settings.training_learning_rate,
                gamma=self.settings.training_gamma,
                gae_lambda=self.settings.training_gae_lambda,
                clip_epsilon=self.settings.training_clip_epsilon,
                entropy_coef=self.settings.training_entropy_coef,
                value_coef=self.settings.training_value_coef,
                prediction_coef=self.settings.training_prediction_coef,
            )
            
            self.agent = PPOAgent(self.model, config)
            
            self.env = TradingEnvironment(
                sequence_length=self.settings.model_sequence_length,
                lot_size=self.settings.trading_lot_size,
                stop_loss_usd=self.settings.trading_stop_loss_usd,
                take_profit_usd=self.settings.trading_take_profit_usd,
                max_loss_usd=self.settings.trading_max_loss_usd,
            )
            
            logger.info("Agent initialized")
    
    async def start(self) -> bool:
        """Start live trading."""
        if self._running:
            return False
        
        self.initialize()
        
        # Initialize Match-Trader client
        self.client = MatchTraderClient(self.settings)
        connected = await self.client.connect()
        
        if not connected:
            self.state.status = "error"
            self.state.error_message = "Failed to connect to Match-Trader API"
            return False
        
        self._running = True
        self.state.status = "running"
        self.state.started_at = datetime.now()
        self.state.error_message = None
        
        # Start trading loop in background
        self._task = asyncio.create_task(self._trading_loop())
        
        logger.info("Live trading started")
        return True
    
    async def stop(self) -> bool:
        """Stop live trading."""
        if not self._running:
            return False
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        if self.client:
            await self.client.disconnect()
        
        self.state.status = "stopped"
        logger.info("Live trading stopped")
        return True
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        try:
            # Reset environment
            obs, info = self.env.reset()
            self.agent.reset_hidden()
            
            while self._running:
                # Get new candle
                candle = await self.client.get_candle(
                    self.settings.trading_symbol,
                    self.settings.trading_timeframe,
                )
                
                if candle is None:
                    await asyncio.sleep(1)
                    continue
                
                self.env.receive_candle(candle)
                
                # Get action
                action, action_info = self.agent.get_action(obs, deterministic=False)
                
                # Execute action
                env_action = {
                    "prediction": action_info["prediction"],
                    "trading_action": action_info["trading_action"],
                }
                
                next_obs, reward, terminated, truncated, step_info = self.env.step(env_action)
                
                # Update state
                self.state.total_steps += 1
                self.state.total_pnl = step_info.get("total_pnl", 0)
                self.state.last_prediction = action_info["prediction"].tolist()
                self.state.last_action = step_info.get("trading_action", "NONE")
                
                if step_info.get("has_position"):
                    self.state.current_position = step_info.get("position_details")
                else:
                    self.state.current_position = None
                
                # Notify callbacks
                update = {
                    "type": "step",
                    "candle": candle.to_dict(),
                    "prediction": action_info["prediction"].tolist(),
                    "action": action_info["trading_action"],
                    "reward": reward,
                    "pnl": self.state.total_pnl,
                    "position": self.state.current_position,
                }
                for callback in self._callbacks:
                    callback(update)
                
                if terminated:
                    # Reset on failure
                    obs, info = self.env.reset()
                    self.agent.reset_hidden()
                else:
                    obs = next_obs
                    self.agent.detach_hidden()
                
                # Wait for next candle (1 minute)
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error("Trading loop error", error=str(e))
            self.state.status = "error"
            self.state.error_message = str(e)
    
    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback for updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_state(self) -> dict:
        """Get current agent state."""
        return {
            "status": self.state.status,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "total_steps": self.state.total_steps,
            "total_trades": self.state.total_trades,
            "total_pnl": self.state.total_pnl,
            "current_position": self.state.current_position,
            "last_prediction": self.state.last_prediction,
            "last_action": self.state.last_action,
            "error_message": self.state.error_message,
        }


class TrainingManager:
    """Manages training lifecycle."""
    
    def __init__(self) -> None:
        """Initialize the training manager."""
        self.settings = get_settings()
        
        self.trainer: Trainer | None = None
        self.status = TrainingStatus()
        self._task: asyncio.Task | None = None
        self._callbacks: list[Callable[[dict], None]] = []
    
    async def start_pretraining(
        self,
        csv_path: str,
        total_timesteps: int = 1_000_000,
    ) -> bool:
        """Start pre-training on CSV data."""
        if self.status.status in ["pretraining", "online_training"]:
            return False
        
        try:
            # Initialize components
            model = HybridActorCritic(
                sequence_length=self.settings.model_sequence_length,
                hidden_dim=self.settings.model_hidden_size,
                num_layers=self.settings.model_num_layers,
                num_heads=self.settings.model_attention_heads,
                dropout=self.settings.model_dropout,
                device=self.settings.device,
            )
            
            agent = PPOAgent(model)
            
            env = TradingEnvironment(
                sequence_length=self.settings.model_sequence_length,
                lot_size=self.settings.trading_lot_size,
                stop_loss_usd=self.settings.trading_stop_loss_usd,
                take_profit_usd=self.settings.trading_take_profit_usd,
                max_loss_usd=self.settings.trading_max_loss_usd,
            )
            
            config = TrainingConfig(
                total_timesteps=total_timesteps,
                checkpoint_dir=self.settings.checkpoint_dir,
                tensorboard_dir=self.settings.tensorboard_dir,
            )
            
            self.trainer = Trainer(agent, env, config)
            
            # Load data
            loader = CSVDataLoader(csv_path)
            loader.load()
            
            self.status.status = "pretraining"
            self.status.started_at = datetime.now()
            self.status.error_message = None
            
            # Run training in background
            self._task = asyncio.create_task(
                self._run_training(loader)
            )
            
            logger.info("Pre-training started", csv_path=csv_path)
            return True
            
        except Exception as e:
            logger.error("Failed to start pre-training", error=str(e))
            self.status.status = "error"
            self.status.error_message = str(e)
            return False
    
    async def _run_training(self, data_source: CSVDataLoader) -> None:
        """Run training in background."""
        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            
            def training_callback(state: TrainingState) -> None:
                self.status.timesteps = state.timesteps
                self.status.episodes = state.episodes
                self.status.best_reward = state.best_reward
                self.status.progress = state.timesteps / self.trainer.config.total_timesteps
                
                # Notify callbacks
                update = {
                    "type": "training_progress",
                    "timesteps": state.timesteps,
                    "episodes": state.episodes,
                    "best_reward": state.best_reward,
                    "progress": self.status.progress,
                }
                for callback in self._callbacks:
                    callback(update)
            
            await loop.run_in_executor(
                None,
                lambda: self.trainer.train(data_source, training_callback),
            )
            
            self.status.status = "idle"
            logger.info("Pre-training completed")
            
        except Exception as e:
            logger.error("Training error", error=str(e))
            self.status.status = "error"
            self.status.error_message = str(e)
    
    async def stop_training(self) -> bool:
        """Stop current training."""
        if self.trainer is None:
            return False
        
        self.trainer.stop()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.status.status = "idle"
        logger.info("Training stopped")
        return True
    
    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback for updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_status(self) -> dict:
        """Get current training status."""
        return {
            "status": self.status.status,
            "progress": self.status.progress,
            "timesteps": self.status.timesteps,
            "episodes": self.status.episodes,
            "best_reward": self.status.best_reward,
            "started_at": self.status.started_at.isoformat() if self.status.started_at else None,
            "error_message": self.status.error_message,
        }


# Global instances (singleton pattern)
_agent_manager: AgentManager | None = None
_training_manager: TrainingManager | None = None


def get_agent_manager() -> AgentManager:
    """Get the agent manager singleton."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager


def get_training_manager() -> TrainingManager:
    """Get the training manager singleton."""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager


async def cleanup_managers() -> None:
    """Cleanup managers on shutdown."""
    global _agent_manager, _training_manager
    
    if _agent_manager:
        await _agent_manager.stop()
    
    if _training_manager:
        await _training_manager.stop_training()

