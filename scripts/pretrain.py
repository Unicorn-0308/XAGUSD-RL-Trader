#!/usr/bin/env python3
"""Pre-training script for CSV data."""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.models.actor_critic import HybridActorCritic
from src.agent.ppo_agent import PPOAgent, PPOConfig
from src.agent.trainer import Trainer, TrainingConfig
from src.environment.trading_env import TradingEnvironment
from src.data.csv_loader import CSVDataLoader
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Pre-train the trading agent on CSV data")
    parser.add_argument(
        "--csv", "-c",
        type=Path,
        required=True,
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )
    parser.add_argument(
        "--checkpoint", "-p",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = get_logger(__name__)
    
    # Get settings
    settings = get_settings()
    
    # Resolve CSV path
    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = settings.historical_data_dir / csv_path
    
    if not csv_path.exists():
        logger.error("CSV file not found", path=str(csv_path))
        sys.exit(1)
    
    logger.info("Starting pre-training", csv_path=str(csv_path), timesteps=args.timesteps)
    
    # Initialize model
    model = HybridActorCritic(
        sequence_length=settings.model_sequence_length,
        hidden_dim=settings.model_hidden_size,
        num_layers=settings.model_num_layers,
        num_heads=settings.model_attention_heads,
        dropout=settings.model_dropout,
        device=settings.device,
    )
    
    # PPO config
    ppo_config = PPOConfig(
        learning_rate=settings.training_learning_rate,
        gamma=settings.training_gamma,
        gae_lambda=settings.training_gae_lambda,
        clip_epsilon=settings.training_clip_epsilon,
        entropy_coef=settings.training_entropy_coef,
        value_coef=settings.training_value_coef,
        prediction_coef=settings.training_prediction_coef,
        batch_size=settings.training_batch_size,
        num_epochs=settings.training_num_epochs,
    )
    
    agent = PPOAgent(model, ppo_config)
    
    # Environment
    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        stop_loss_usd=settings.trading_stop_loss_usd,
        take_profit_usd=settings.trading_take_profit_usd,
        max_loss_usd=settings.trading_max_loss_usd,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
        checkpoint_mgr.load(model, agent.optimizer, checkpoint_path=args.checkpoint)
        logger.info("Loaded checkpoint", path=str(args.checkpoint))
    
    # Training config
    train_config = TrainingConfig(
        total_timesteps=args.timesteps,
        checkpoint_dir=settings.checkpoint_dir,
        tensorboard_dir=settings.tensorboard_dir,
        checkpoint_interval_minutes=settings.training_checkpoint_interval_minutes,
    )
    
    trainer = Trainer(agent, env, train_config)
    
    # Load data
    loader = CSVDataLoader(csv_path)
    loader.load()
    logger.info("Loaded CSV data", num_candles=len(loader))
    
    # Train
    try:
        state = trainer.train(loader)
        
        logger.info(
            "Training completed",
            timesteps=state.timesteps,
            episodes=state.episodes,
            best_reward=state.best_reward,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()

