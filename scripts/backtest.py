#!/usr/bin/env python3
"""Backtesting script for evaluating trained models."""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.models.actor_critic import HybridActorCritic
from src.agent.ppo_agent import PPOAgent
from src.environment.trading_env import TradingEnvironment
from src.data.csv_loader import CSVDataLoader
from src.utils.checkpoint import CheckpointManager
from src.utils.logging import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Backtest a trained model on historical data")
    parser.add_argument(
        "--csv", "-c",
        type=Path,
        required=True,
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--checkpoint", "-p",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--deterministic", "-d",
        action="store_true",
        help="Use deterministic actions",
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
    
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found", path=str(args.checkpoint))
        sys.exit(1)
    
    logger.info("Starting backtest", csv_path=str(csv_path), checkpoint=str(args.checkpoint))
    
    # Initialize model
    model = HybridActorCritic(
        sequence_length=settings.model_sequence_length,
        hidden_dim=settings.model_hidden_size,
        num_layers=settings.model_num_layers,
        num_heads=settings.model_attention_heads,
        device=settings.device,
    )
    
    agent = PPOAgent(model)
    
    # Environment
    env = TradingEnvironment(
        sequence_length=settings.model_sequence_length,
        lot_size=settings.trading_lot_size,
        stop_loss_usd=settings.trading_stop_loss_usd,
        take_profit_usd=settings.trading_take_profit_usd,
        max_loss_usd=settings.trading_max_loss_usd,
    )
    
    # Load checkpoint
    checkpoint_mgr = CheckpointManager(settings.checkpoint_dir)
    checkpoint_mgr.load(model, checkpoint_path=args.checkpoint)
    logger.info("Loaded checkpoint")
    
    # Load data
    loader = CSVDataLoader(csv_path)
    loader.load()
    logger.info("Loaded CSV data", num_candles=len(loader))
    
    # Run backtest
    candles = loader.candles
    initial_candles = candles[:settings.model_sequence_length]
    
    obs, _ = env.reset(options={"initial_candles": initial_candles})
    agent.reset_hidden()
    
    total_reward = 0.0
    steps = 0
    trade_log = []
    
    logger.info("Running backtest...")
    
    for i, candle in enumerate(candles[settings.model_sequence_length:]):
        env.receive_candle(candle)
        
        action, info = agent.get_action(obs, deterministic=args.deterministic)
        
        env_action = {
            "prediction": info["prediction"],
            "trading_action": info["trading_action"],
        }
        
        obs, reward, terminated, truncated, step_info = env.step(env_action)
        
        total_reward += reward
        steps += 1
        
        # Log trades
        if "realized_pnl" in step_info:
            trade_log.append({
                "step": steps,
                "action": step_info.get("trading_action"),
                "pnl": step_info.get("realized_pnl"),
            })
        
        if terminated:
            logger.warning("Episode terminated early (max loss reached)")
            break
        
        # Progress update every 10%
        progress = (i + 1) / (len(candles) - settings.model_sequence_length)
        if (i + 1) % max(1, (len(candles) - settings.model_sequence_length) // 10) == 0:
            logger.info(f"Progress: {progress:.1%}, PnL: {step_info.get('total_pnl', 0):.2f}")
    
    # Get results
    stats = env.get_statistics()
    trading_stats = stats["trading_stats"]
    
    results = {
        "backtest_info": {
            "csv_file": str(csv_path),
            "checkpoint": str(args.checkpoint),
            "total_candles": len(candles),
            "tested_candles": steps,
        },
        "performance": {
            "steps": steps,
            "total_reward": total_reward,
            "avg_reward_per_step": total_reward / max(steps, 1),
        },
        "trading_stats": trading_stats,
        "trades": trade_log,
    }
    
    # Print results
    logger.info("=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Steps: {steps}")
    logger.info(f"Total Reward: {total_reward:.4f}")
    logger.info(f"Total PnL: ${trading_stats['total_pnl']:.2f}")
    logger.info(f"Total Trades: {trading_stats['total_trades']}")
    logger.info(f"Win Rate: {trading_stats['win_rate']:.2%}")
    logger.info(f"Profit Factor: {trading_stats['profit_factor']:.2f}")
    logger.info(f"Avg Win: ${trading_stats['avg_win']:.2f}")
    logger.info(f"Avg Loss: ${trading_stats['avg_loss']:.2f}")
    logger.info("=" * 50)
    
    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

