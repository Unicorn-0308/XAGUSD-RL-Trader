#!/usr/bin/env python3
"""Live trading script (connects to Match-Trader API)."""

import argparse
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.models.actor_critic import HybridActorCritic
from src.agent.ppo_agent import PPOAgent, PPOConfig
from src.environment.trading_env import TradingEnvironment
from src.data.match_trader_client import MatchTraderClient
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import setup_logging, get_logger


async def main():
    parser = argparse.ArgumentParser(description="Run live trading with Match-Trader API")
    parser.add_argument(
        "--checkpoint", "-p",
        type=Path,
        default=None,
        help="Path to model checkpoint (uses latest if not specified)",
    )
    parser.add_argument(
        "--best", "-b",
        action="store_true",
        help="Use best checkpoint instead of latest",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo mode (simulated trading)",
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
    
    logger.info("Initializing live trading...")
    
    # Initialize model
    model = HybridActorCritic(
        sequence_length=settings.model_sequence_length,
        hidden_dim=settings.model_hidden_size,
        num_layers=settings.model_num_layers,
        num_heads=settings.model_attention_heads,
        device=settings.device,
    )
    
    ppo_config = PPOConfig(
        learning_rate=settings.training_learning_rate,
    )
    
    agent = PPOAgent(model, ppo_config)
    
    # Load checkpoint
    checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
    
    if args.checkpoint:
        checkpoint_mgr.load(model, checkpoint_path=args.checkpoint)
        logger.info("Loaded checkpoint", path=str(args.checkpoint))
    elif args.best:
        best = checkpoint_mgr.get_best_checkpoint()
        if best:
            checkpoint_mgr.load(model, load_best=True)
            logger.info("Loaded best checkpoint")
        else:
            logger.error("No best checkpoint found")
            sys.exit(1)
    else:
        latest = checkpoint_mgr.get_latest_checkpoint()
        if latest:
            checkpoint_mgr.load(model)
            logger.info("Loaded latest checkpoint")
        else:
            logger.warning("No checkpoint found, using untrained model")
    
    # Environment
    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        stop_loss_usd=settings.trading_stop_loss_usd,
        take_profit_usd=settings.trading_take_profit_usd,
        max_loss_usd=settings.trading_max_loss_usd,
    )
    
    # Connect to Match-Trader API
    client = MatchTraderClient(settings)
    connected = await client.connect()
    
    if not connected:
        logger.error("Failed to connect to Match-Trader API")
        sys.exit(1)
    
    logger.info("Connected to Match-Trader API")
    
    try:
        # Get initial candles
        initial_candles = await client.get_candles(
            settings.trading_symbol,
            settings.trading_timeframe,
            settings.model_sequence_length,
        )
        
        if len(initial_candles) < settings.model_sequence_length:
            logger.error("Not enough historical data")
            sys.exit(1)
        
        # Reset environment
        obs, _ = env.reset(options={"initial_candles": initial_candles})
        agent.reset_hidden()
        
        logger.info("Starting live trading loop...")
        
        while True:
            # Get new candle
            candle = await client.get_candle(
                settings.trading_symbol,
                settings.trading_timeframe,
            )
            
            if candle is None:
                await asyncio.sleep(1)
                continue
            
            env.receive_candle(candle)
            
            # Get action
            action, info = agent.get_action(obs, deterministic=False)
            
            # Execute action
            env_action = {
                "prediction": info["prediction"],
                "trading_action": info["trading_action"],
            }
            
            obs, reward, terminated, truncated, step_info = env.step(env_action)
            
            # Log
            logger.info(
                "Step completed",
                action=step_info.get("trading_action"),
                pnl=step_info.get("total_pnl"),
                prediction=info["prediction"].tolist(),
            )
            
            if terminated:
                logger.warning("Episode terminated (max loss reached), resetting...")
                initial_candles = await client.get_candles(
                    settings.trading_symbol,
                    settings.trading_timeframe,
                    settings.model_sequence_length,
                )
                obs, _ = env.reset(options={"initial_candles": initial_candles})
                agent.reset_hidden()
            else:
                agent.detach_hidden()
            
            # Wait for next candle
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    finally:
        await client.disconnect()
        logger.info("Disconnected from Match-Trader API")


if __name__ == "__main__":
    asyncio.run(main())

