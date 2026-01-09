"""
Run Production Trader Simulation
==================================
Test the production trader against historical data.

Usage:
    python run_simulation.py --start 2024-01-01 --end 2024-03-31
    python run_simulation.py --start 2024-01-01 --end 2024-03-31 --config config.yaml
    python run_simulation.py --start 2024-01-01 --end 2024-03-31 --log-level SIM  # Clean progress bar
    python run_simulation.py --start 2024-01-01 --end 2024-03-31 --output sim_trades.csv  # Save trades
"""
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from production_trader.config import load_config, validate_config
from production_trader.simulation.replay_engine import ReplayEngine


def setup_logging(level='INFO'):
    """Setup logging for simulation"""
    if level == 'SIM':
        # Minimal logging for clean simulation output
        logging.basicConfig(
            level=logging.CRITICAL,
            format='%(message)s',
            handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )


def main():
    parser = argparse.ArgumentParser(description='Run Production Trader Simulation')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--data-dir', default='data_15m', help='Historical data directory')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'SIM'])
    parser.add_argument('--output', help='Output CSV file for trade results (optional)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("="*100)
    logger.info("PRODUCTION TRADER SIMULATION")
    logger.info("="*100)
    logger.info(f"Start date: {args.start}")
    logger.info(f"End date:   {args.end}")
    logger.info(f"Config:     {args.config}")
    logger.info(f"Data dir:   {args.data_dir}")
    logger.info("="*100)

    try:
        # Load configuration
        quiet = (args.log_level == 'SIM')

        if not quiet:
            logger.info("Loading configuration...")
        config_path = Path(__file__).parent / args.config
        config = load_config(str(config_path))
        validate_config(config)

        # Create and run simulation
        if not quiet:
            logger.info("Initializing replay engine...")
        engine = ReplayEngine(
            config=config,
            start_date=args.start,
            end_date=args.end,
            data_dir=args.data_dir,
            quiet=quiet
        )

        if not quiet:
            logger.info("Running simulation...")
        results = engine.run()

        # Display final summary
        print("\n" + "="*100)
        print("SIMULATION COMPLETE")
        print("="*100)
        print(f"Period:          {results['start_date']} to {results['end_date']}")
        print(f"Duration:        {results['duration_years']:.2f} years")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital:   ${results['final_capital']:.2f}")
        print(f"Total Return:    {results['total_return']:.1%}")
        print(f"CAGR:            {results['cagr']:.1%}")
        print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
        print()
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Win Rate:        {results['win_rate']:.1%}")
        print(f"Winners:         {results['winning_trades']}")
        print(f"Losers:          {results['losing_trades']}")
        print()
        print(f"Signals Generated: {results['signals_generated']}")
        print(f"Trades Opened:     {results['trades_opened']}")
        print(f"Trades Closed:     {results['trades_closed']}")
        print("="*100)

        # Save trades to CSV if output specified
        if args.output and results['trades']:
            df = pd.DataFrame(results['trades'])
            df.to_csv(args.output, index=False)
            print(f"\nTrades saved to: {args.output}")
            print(f"Total trades exported: {len(df)}")

        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
