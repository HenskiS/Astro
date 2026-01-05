"""
Production Trading System - Main Orchestrator
==============================================
Coordinates all trading components and runs the main event loop.

Usage:
    python main.py [--config config.yaml] [--dry-run]
"""
import sys
import os
import time
import signal
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from production_trader.config import load_config, validate_config
from production_trader.execution.oanda_broker import OandaBroker
from production_trader.execution.position_manager import PositionManager
from production_trader.state.state_manager import StateManager
from production_trader.strategies.strategy_15m import Strategy15m
from production_trader.risk.risk_manager import RiskManager
from production_trader.monitoring.telegram_notifier import TelegramNotifier


# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    print("\n🛑 Shutdown signal received. Closing gracefully...")
    running = False


def setup_logging(config):
    """Setup logging configuration"""
    log_file = Path(config.monitoring.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.monitoring.logging.level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def should_check_15m_signals(current_time: datetime, last_check: str = None) -> bool:
    """
    Check if we should generate 15m signals.

    Args:
        current_time: Current datetime
        last_check: ISO timestamp of last check

    Returns:
        True if should check (at :00, :15, :30, :45 within first 5 seconds)
    """
    # Only check during the first 5 seconds of the 15-minute marks
    if current_time.minute not in [0, 15, 30, 45]:
        return False

    if current_time.second > 5:
        return False

    if last_check is None:
        return True

    # Check if we're in a NEW 15-minute window (not the same one as last check)
    last_check_time = datetime.fromisoformat(last_check)

    # Get the 15-minute window for both times (0, 15, 30, 45)
    current_window = (current_time.hour * 60 + current_time.minute) // 15
    last_window = (last_check_time.hour * 60 + last_check_time.minute) // 15

    return current_window != last_window


def get_seconds_until_next_15m_mark(current_time: datetime) -> int:
    """
    Calculate seconds until the next 15-minute mark.

    Args:
        current_time: Current datetime

    Returns:
        Seconds to wait until next :00, :15, :30, or :45 minute
    """
    current_minute = current_time.minute
    current_second = current_time.second

    # Find next 15-minute mark
    if current_minute < 15:
        next_mark = 15
    elif current_minute < 30:
        next_mark = 30
    elif current_minute < 45:
        next_mark = 45
    else:
        next_mark = 60  # Will roll to next hour

    # Calculate seconds to wait
    minutes_to_wait = next_mark - current_minute
    seconds_to_wait = (minutes_to_wait * 60) - current_second

    return seconds_to_wait


def should_update_positions(current_time: datetime, last_check: str = None) -> bool:
    """
    Check if we should update positions (every minute).

    Args:
        current_time: Current datetime
        last_check: ISO timestamp of last check

    Returns:
        True if should update (within first 5 seconds of each minute)
    """
    # Only update within first 5 seconds of each minute
    if current_time.second > 5:
        return False

    if last_check is None:
        return True

    # Check if we're in a NEW minute (not the same one as last check)
    last_check_time = datetime.fromisoformat(last_check)

    # Compare hour and minute (ignore seconds)
    current_minute_mark = (current_time.hour * 60 + current_time.minute)
    last_minute_mark = (last_check_time.hour * 60 + last_check_time.minute)

    return current_minute_mark != last_minute_mark


def main():
    """Main trading loop"""
    global running

    # Parse arguments
    parser = argparse.ArgumentParser(description='Production Trading System')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no actual trades)')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    validate_config(config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("PRODUCTION TRADING SYSTEM STARTING")
    logger.info("="*80)
    logger.info(f"Account type: {config.oanda.account_type}")
    logger.info(f"Initial capital: ${config.capital.initial}")
    logger.info(f"Strategy: 15m breakout")
    logger.info(f"Pairs: {', '.join(config.strategy_15m.pairs)}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*80)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components
    try:
        logger.info("Initializing broker...")
        broker = OandaBroker(
            api_key=config.oanda.api_key,
            account_id=config.oanda.account_id,
            account_type=config.oanda.account_type
        )

        if not broker.check_connection():
            logger.error("Failed to connect to OANDA")
            return 1

        # Get actual account balance from OANDA
        logger.info("Fetching account balance...")
        account = broker.get_account_summary()
        if not account:
            logger.error("Failed to get account summary")
            return 1

        actual_balance = account.balance
        logger.info(f"OANDA account balance: ${actual_balance:.2f}")

        logger.info("Initializing state manager...")
        state_manager = StateManager(config.state.json_file)

        # If this is a fresh start (no existing state), use actual OANDA balance
        state_file = Path(config.state.json_file)
        if not state_file.exists():
            logger.info(f"Fresh start - syncing capital with OANDA balance: ${actual_balance:.2f}")
            state_manager.set_capital(actual_balance)
            state_manager.save_state()
        else:
            logger.info(f"Loaded existing state - capital: ${state_manager.get_capital():.2f}")

        logger.info("Initializing strategy...")
        strategy = Strategy15m(config.strategy_15m, broker)

        logger.info("Initializing position manager...")
        position_manager = PositionManager(config.strategy_15m, broker, state_manager)

        # Reconcile positions at startup (sync local state with OANDA)
        position_manager.reconcile_positions_at_startup()

        logger.info("Initializing risk manager...")
        risk_manager = RiskManager(config.capital, state_manager)

        logger.info("Initializing Telegram notifier...")
        telegram = TelegramNotifier(config.monitoring.telegram)

        # Send startup notification
        telegram.notify_system_start({
            'account_type': config.oanda.account_type,
            'capital': state_manager.get_capital(),
            'pairs_count': len(config.strategy_15m.pairs),
            'max_positions': config.strategy_15m.max_positions_total
        })

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1

    # Main event loop
    logger.info("Entering main trading loop...")
    print("\n✓ System running. Press Ctrl+C to stop.\n")

    loop_count = 0

    while running:
        try:
            loop_count += 1
            current_time = datetime.now()

            # Check for KILL_SWITCH file
            if risk_manager.check_kill_switch():
                logger.critical("KILL_SWITCH file detected! Emergency shutdown initiated.")
                print("\n🚨 KILL SWITCH ACTIVATED - SYSTEM HALTING")
                telegram.notify_api_error("KILL_SWITCH activated - emergency shutdown")
                position_manager.close_all_positions('kill_switch')
                running = False
                continue

            # Update positions (every minute)
            last_position_update = state_manager.state.get('last_position_update')
            if should_update_positions(current_time, last_position_update) or loop_count == 1:
                open_count = position_manager.get_position_count()
                if open_count > 0:
                    logger.info(f"Checking {open_count} open position(s)...")
                closed_count = position_manager.update_all_positions()
                if closed_count > 0:
                    logger.info(f"Updated positions, closed {closed_count}")
                elif open_count > 0:
                    logger.debug(f"All {open_count} position(s) still open")
                state_manager.update_last_check('position_update')

            # Check 15m signals (at :00, :15, :30, :45)
            last_15m_check = state_manager.state.get('last_15m_check')
            if should_check_15m_signals(current_time, last_15m_check):
                logger.info(f"Checking 15m signals at {current_time.strftime('%H:%M')}")

                # Check if trading is allowed
                if not risk_manager.can_trade():
                    logger.warning("Trading disabled - skipping signal generation")
                    state_manager.update_last_check('15m_check')
                    continue

                # Generate signals
                current_capital = state_manager.get_capital()
                existing_positions = position_manager.get_open_positions_by_pair()

                signals = strategy.generate_signals(current_capital, existing_positions)

                # Process signals
                for sig in signals:
                    # Check position limits
                    if not risk_manager.check_position_limits(
                        sig, existing_positions, config.strategy_15m
                    ):
                        logger.debug(f"Position limits exceeded for {sig['pair']}")
                        continue

                    # Validate position size
                    prices = broker.get_current_prices([sig['pair']])
                    if not prices:
                        continue

                    price_data = prices[sig['pair']]
                    validated_size = risk_manager.validate_position_size(
                        sig, current_capital, price_data.mid_close
                    )

                    if validated_size is None:
                        logger.warning(f"Invalid position size for {sig['pair']}")
                        continue

                    sig['size'] = validated_size

                    # Open position
                    if position_manager.open_position(sig):
                        telegram.notify_position_opened({
                            'pair': sig['pair'],
                            'direction': sig['direction'],
                            'size': sig['size'],
                            'entry_price': price_data.mid_close,
                            'confidence': sig['confidence']
                        })

                state_manager.update_last_check('15m_check')

            # Check emergency conditions (every 5 minutes)
            last_emergency_check = state_manager.state.get('last_emergency_check')
            if current_time.minute % 5 == 0 and current_time.second <= 5:
                # Prevent duplicate checks within same 5-minute window
                should_check = True
                if last_emergency_check:
                    last_check_time = datetime.fromisoformat(last_emergency_check)
                    # Use window-based comparison (same as signal checks)
                    current_window = (current_time.hour * 60 + current_time.minute) // 5
                    last_window = (last_check_time.hour * 60 + last_check_time.minute) // 5
                    if current_window == last_window:
                        should_check = False

                if should_check:
                    logger.debug("Checking emergency conditions...")

                    # Get account summary
                    account = broker.get_account_summary()
                    if account:
                        current_capital = account.balance
                        state_manager.set_capital(current_capital)

                        # Check max drawdown
                        if risk_manager.check_max_drawdown(current_capital):
                            logger.critical("MAX DRAWDOWN EXCEEDED - EMERGENCY SHUTDOWN")
                            print("\n🚨 EMERGENCY: Max drawdown exceeded - closing all positions")

                            # Close all positions
                            position_manager.close_all_positions('max_drawdown')

                            # Send Telegram alert
                            telegram.notify_max_drawdown(
                                current_capital,
                                state_manager.get_peak_capital()
                            )

                            running = False
                            continue

                        # Check daily loss limit
                        if risk_manager.check_daily_loss_limit():
                            logger.critical("DAILY LOSS LIMIT EXCEEDED")
                            telegram.notify_daily_loss_limit(
                                state_manager.get_daily_pnl(),
                                current_capital
                            )

                    state_manager.update_last_check('emergency_check')

            # Reset safe mode at start of new day (00:00 UTC)
            last_daily_reset = state_manager.state.get('last_daily_reset')
            if current_time.hour == 0 and current_time.minute == 0 and current_time.second <= 5:
                # Prevent duplicate resets within same day
                should_reset = True
                if last_daily_reset:
                    last_reset_time = datetime.fromisoformat(last_daily_reset)
                    if (current_time - last_reset_time) < timedelta(hours=23):
                        should_reset = False

                if should_reset:
                    logger.info("New trading day - resetting safe mode")
                    risk_manager.reset_safe_mode()

                    # Send daily summary
                    telegram.send_daily_summary(
                        capital=state_manager.get_capital(),
                        daily_pnl=state_manager.get_daily_pnl(),
                        trades_today=state_manager.state.get('trades_today', 0),
                        open_positions=position_manager.get_position_count()
                    )

                    state_manager.update_last_check('daily_reset')

            # Save state (every 15 minutes)
            last_state_save = state_manager.state.get('last_state_save')
            if current_time.minute in [0, 15, 30, 45] and current_time.second <= 5:
                # Prevent duplicate saves within same 15-minute window
                should_save = True
                if last_state_save:
                    last_save_time = datetime.fromisoformat(last_state_save)
                    # Use window-based comparison (same as signal checks)
                    current_window = (current_time.hour * 60 + current_time.minute) // 15
                    last_window = (last_save_time.hour * 60 + last_save_time.minute) // 15
                    if current_window == last_window:
                        should_save = False

                if should_save:
                    state_manager.save_state()
                    state_manager.state['last_state_save'] = current_time.isoformat()
                    logger.debug("State saved")

            # Status update (every hour)
            last_status_update = state_manager.state.get('last_status_update')
            if current_time.minute == 0 and current_time.second <= 5:
                # Prevent duplicate updates within same hour
                should_update = True
                if last_status_update:
                    last_update_time = datetime.fromisoformat(last_status_update)
                    # Use window-based comparison (same as signal checks)
                    current_hour_window = current_time.hour
                    last_hour_window = last_update_time.hour
                    if current_hour_window == last_hour_window:
                        should_update = False

                if should_update:
                    account = broker.get_account_summary()
                    if account:
                        logger.info(f"Status: Capital=${account.balance:.2f}, Open trades={account.open_trade_count}")
                        state_manager.state['last_status_update'] = current_time.isoformat()

            # Heartbeat (every 5 minutes to show system is alive)
            last_heartbeat = state_manager.state.get('last_heartbeat')
            if current_time.minute % 5 == 0 and current_time.second <= 5:
                should_heartbeat = True
                if last_heartbeat:
                    last_heartbeat_time = datetime.fromisoformat(last_heartbeat)
                    current_window = (current_time.hour * 60 + current_time.minute) // 5
                    last_window = (last_heartbeat_time.hour * 60 + last_heartbeat_time.minute) // 5
                    if current_window == last_window:
                        should_heartbeat = False

                if should_heartbeat:
                    capital = state_manager.get_capital()
                    open_positions = position_manager.get_position_count()
                    logger.info(f"Heartbeat | Capital: ${capital:.2f} | Open: {open_positions} | Time: {current_time.strftime('%H:%M')}")
                    state_manager.state['last_heartbeat'] = current_time.isoformat()

            # Smart sleep: wake up just before 15-minute marks
            seconds_until_15m = get_seconds_until_next_15m_mark(current_time)

            # If close to a 15m mark (within 2 minutes), use short sleep
            if seconds_until_15m <= 120:
                sleep_time = 1  # 1 second polling when close to signal time
            else:
                # Otherwise sleep until ~30 seconds before next mark (max 60 seconds for responsiveness)
                sleep_time = min(60, max(1, seconds_until_15m - 30))

            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            running = False

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)  # Wait before retrying

    # Shutdown
    logger.info("="*80)
    logger.info("SHUTTING DOWN")
    logger.info("="*80)

    # Save final state
    state_manager.save_state()
    logger.info("Final state saved")

    # Get final status
    final_capital = state_manager.get_capital()
    total_trades = state_manager.state.get('total_trades', 0)

    try:
        account = broker.get_account_summary()
        if account:
            final_capital = account.balance
            logger.info(f"Final balance: ${account.balance:.2f}")
            logger.info(f"Open trades: {account.open_trade_count}")
    except:
        pass

    # Send shutdown notification
    telegram.notify_system_stop(final_capital, total_trades)

    logger.info("Shutdown complete")
    print("\n✓ System stopped gracefully\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
