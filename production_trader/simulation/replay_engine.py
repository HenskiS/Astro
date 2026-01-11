"""
Replay Engine
=============
Orchestrates the simulation by running production code against historical data.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from production_trader.simulation.mock_broker import MockBroker
from production_trader.simulation.time_simulator import TimeSimulator
from production_trader.execution.position_manager import PositionManager
from production_trader.state.state_manager import StateManager
from production_trader.strategies.strategy_15m import Strategy15m
from production_trader.risk.risk_manager import RiskManager
from production_trader.config import Config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Replays production trading logic against historical data.

    This runs the ACTUAL production code with a MockBroker,
    ensuring the simulation matches real trading behavior.
    """

    def __init__(self, config: Config, start_date: str, end_date: str,
                 data_dir: str = 'data_15m', quiet: bool = False, trained_models: Dict = None,
                 predictions: Dict = None):
        """
        Initialize replay engine.

        Args:
            config: Production configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_dir: Directory with historical CSV files
            quiet: If True, use minimal logging with progress bar
            trained_models: Pre-trained models dict (optional, if None loads from disk)
            predictions: Pre-generated predictions dict (optional)
        """
        self.config = config
        self.data_dir = data_dir
        self.quiet = quiet

        # Parse dates and make timezone-aware (UTC)
        import pytz
        self.official_start_time = datetime.fromisoformat(start_date).replace(hour=0, minute=0)
        self.end_time = datetime.fromisoformat(end_date).replace(hour=23, minute=45)

        # Make timezone-aware if needed
        if self.official_start_time.tzinfo is None:
            self.official_start_time = pytz.UTC.localize(self.official_start_time)
        if self.end_time.tzinfo is None:
            self.end_time = pytz.UTC.localize(self.end_time)

        # Add warmup period (220 bars = 3,300 minutes = 55 hours = ~2.3 days)
        # This gives the strategy enough historical data from day 1
        warmup_bars = 220
        warmup_duration = timedelta(minutes=15 * warmup_bars)
        self.warmup_start_time = self.official_start_time - warmup_duration

        # Initialize time simulator with warmup period
        self.time_sim = TimeSimulator(self.warmup_start_time, self.end_time)
        self.start_time = self.warmup_start_time  # For compatibility

        # Initialize mock broker (starts at warmup time)
        self.broker = MockBroker(
            data_dir=data_dir,
            pairs=config.strategy_15m.pairs,
            initial_balance=config.capital.initial,
            current_time=self.warmup_start_time
        )

        # Create temporary state file for simulation
        sim_state_file = 'simulation_state.json'
        # Delete old state file to start fresh
        import os
        if os.path.exists(sim_state_file):
            os.remove(sim_state_file)
        self.state_manager = StateManager(sim_state_file)
        self.state_manager.set_capital(config.capital.initial)

        # Initialize production components
        self.strategy = Strategy15m(config.strategy_15m, self.broker, models=trained_models, predictions=predictions)
        self.position_manager = PositionManager(config.strategy_15m, self.broker, self.state_manager)
        # Clear any phantom positions loaded from old state files
        self.position_manager.positions = []  # List, not dict!
        logger.info(f"Cleared phantom positions - starting fresh")
        self.risk_manager = RiskManager(config.capital, self.state_manager)

        # Track simulation events
        self.events = []
        self.signals_generated = 0
        self.trades_opened = 0
        self.trades_closed = 0

        # Queue for pending signals (execute at next bar open)
        self.pending_signals = []

        # Track max drawdown
        self.peak_capital = config.capital.initial
        self.max_drawdown = 0.0

        if not quiet:
            logger.info("="*80)
            logger.info("SIMULATION REPLAY ENGINE INITIALIZED")
            logger.info("="*80)
            logger.info(f"Test period: {start_date} to {end_date}")
            logger.info(f"Warmup period: {self.warmup_start_time.date()} (220 bars for feature calculation)")
            logger.info(f"Initial capital: ${config.capital.initial:.2f}")
            logger.info(f"Pairs: {', '.join(config.strategy_15m.pairs)}")
            logger.info("="*80)
        else:
            print(f"\nSimulating {start_date} to {end_date} (with warmup from {self.warmup_start_time.date()})...")
            print(f"Initial capital: ${config.capital.initial:.2f}\n")

    def run(self) -> Dict:
        """
        Run the simulation bar-by-bar.

        Returns:
            Dict with simulation results
        """
        if not self.quiet:
            logger.info("Starting simulation...")

        bar_count = 0
        last_day = None

        # Calculate total bars for progress bar
        total_bars = self.time_sim.get_remaining_bars() + 1
        last_progress_pct = 0

        # Use tqdm progress bar in quiet mode
        if self.quiet and HAS_TQDM:
            pbar = tqdm(total=total_bars, desc="Simulating", unit="bar",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = None

        while True:
            bar_count += 1

            # Update broker's current time
            self.broker.set_current_time(self.time_sim.current_time)

            # Debug: Log simulation time at start of each bar
            if bar_count % 4 == 1:  # Log every hour (4 bars = 1 hour)
                logger.debug(f"[Bar {bar_count}] Simulation time: {self.time_sim.current_time}")

            # Execute pending signals at this bar's open price
            if self.pending_signals:
                self._execute_pending_signals()

            # Check for new day (reset daily stats)
            if self.time_sim.is_new_day(last_day):
                if last_day is not None:
                    if not self.quiet:
                        logger.info(f"New day: {self.time_sim.current_time.date()}")
                    self.risk_manager.reset_safe_mode()
                last_day = self.time_sim.current_time

            # Update existing positions (every bar)
            if self.position_manager.get_position_count() > 0:
                closed_count = self.position_manager.update_all_positions()
                if closed_count > 0:
                    # Only count closes during the official test period
                    if self.time_sim.current_time >= self.official_start_time:
                        self.trades_closed += closed_count
                    if not self.quiet:
                        logger.info(f"[{self.time_sim.current_time}] Closed {closed_count} position(s)")

            # Track drawdown (every bar)
            current_capital = self.state_manager.get_capital()
            if current_capital > self.peak_capital:
                self.peak_capital = current_capital
            drawdown = (self.peak_capital - current_capital) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

            # Update progress bar AFTER positions update and balance changes
            if pbar:
                pbar.update(1)
                # Update equity display every 4 bars (every hour)
                if bar_count % 4 == 0:
                    current_equity = self.broker.balance
                    equity_pct = ((current_equity - self.config.capital.initial) / self.config.capital.initial) * 100
                    pbar.set_description(f"Simulating [${current_equity:.2f} ({equity_pct:+.1f}%)]")
            elif self.quiet:
                # Simple progress indicator when tqdm not available
                progress_pct = int((bar_count / total_bars) * 100)
                if progress_pct >= last_progress_pct + 5:  # Update every 5%
                    elapsed = self.time_sim.current_time - self.start_time
                    remaining_bars = total_bars - bar_count
                    avg_time_per_bar = elapsed.total_seconds() / bar_count
                    eta_seconds = remaining_bars * avg_time_per_bar
                    eta_mins = int(eta_seconds / 60)
                    current_equity = self.broker.balance
                    equity_pct = ((current_equity - self.config.capital.initial) / self.config.capital.initial) * 100
                    print(f"Progress: {progress_pct}% | Bars: {bar_count}/{total_bars} | ETA: {eta_mins}m | Equity: ${current_equity:.2f} ({equity_pct:+.1f}%)", end='\r', flush=True)
                    last_progress_pct = progress_pct

            # Check for 15-minute signals
            if self.time_sim.is_15m_signal_time():
                self._check_signals()

            # Emergency checks (drawdown, daily loss limit)
            if self.time_sim.is_emergency_check_time():
                self._check_emergency_conditions()

            # Progress updates (non-quiet mode)
            if not self.quiet and bar_count % 100 == 0:
                progress = self.time_sim.get_progress_pct()
                capital = self.state_manager.get_capital()
                open_pos = self.position_manager.get_position_count()
                logger.info(f"Progress: {progress:.1f}% | Capital: ${capital:.2f} | Open: {open_pos} | Bar {bar_count}")

            # Advance to next bar
            if not self.time_sim.advance_to_next_bar():
                break  # Simulation complete

        # Close progress bar
        if pbar:
            pbar.close()
        elif self.quiet:
            print()  # Newline after progress indicator

        if not self.quiet:
            logger.info("Simulation complete!")
        return self._get_results()

    def _check_signals(self):
        """Check for trading signals (called at 15-minute marks)"""
        # Check if trading is allowed
        if not self.risk_manager.can_trade():
            return

        # Generate signals
        current_capital = self.state_manager.get_capital()
        existing_positions = self.position_manager.get_open_positions_by_pair()

        signals = self.strategy.generate_signals(current_capital, existing_positions)
        self.signals_generated += len(signals)

        if signals:
            logger.info(f"[SIM TIME: {self.time_sim.current_time}] Generated {len(signals)} signal(s)")

        # Process signals - queue them for execution at next bar open
        for sig in signals:
            # Check position limits
            if not self.risk_manager.check_position_limits(
                sig, existing_positions, self.config.strategy_15m
            ):
                logger.debug(f"Signal filtered by position limits: {sig['pair']} {sig['direction']}")
                continue

            # Validate position size
            prices = self.broker.get_current_prices([sig['pair']])
            if not prices:
                logger.warning(f"No prices available for {sig['pair']}")
                continue

            price_data = prices[sig['pair']]
            validated_size = self.risk_manager.validate_position_size(
                sig, current_capital, price_data.mid_close
            )

            if validated_size is None:
                logger.warning(f"Position size validation failed for {sig['pair']}")
                continue

            sig['size'] = validated_size

            # Queue signal for execution at next bar open (prevents lookahead bias)
            self.pending_signals.append(sig)
            logger.info(f"[SIM TIME: {self.time_sim.current_time}] Queued {sig['direction']} {sig['pair']} "
                      f"conf={sig['confidence']:.2f} (will execute at next bar open)")

    def _execute_pending_signals(self):
        """Execute pending signals at current bar's open price"""
        executed_count = 0
        in_test_period = self.time_sim.current_time >= self.official_start_time

        for sig in self.pending_signals:
            # Open position using open price (use_open_price=True)
            if self.position_manager.open_position(sig, use_open_price=True):
                # Only count trades during the official test period (not warmup)
                if in_test_period:
                    self.trades_opened += 1
                executed_count += 1
                logger.info(f"[SIM TIME: {self.time_sim.current_time}] Executed {sig['direction']} {sig['pair']} "
                          f"at bar open | conf={sig['confidence']:.2f}{' [WARMUP]' if not in_test_period else ''}")

        # Clear pending signals
        self.pending_signals = []

        if executed_count > 0:
            logger.info(f"[SIM TIME: {self.time_sim.current_time}] Executed {executed_count} pending signal(s)")

    def _check_emergency_conditions(self):
        """Check for emergency conditions (drawdown, loss limits)"""
        account = self.broker.get_account_summary()
        if not account:
            return

        current_capital = account.balance
        self.state_manager.set_capital(current_capital)

        # Check max drawdown
        if self.risk_manager.check_max_drawdown(current_capital):
            if not self.quiet:
                logger.critical(f"[{self.time_sim.current_time}] MAX DRAWDOWN EXCEEDED!")
            self.position_manager.close_all_positions('max_drawdown')
            return

        # Check daily loss limit
        if self.risk_manager.check_daily_loss_limit():
            if not self.quiet:
                logger.warning(f"[{self.time_sim.current_time}] Daily loss limit exceeded")

    def _get_results(self) -> Dict:
        """Get simulation results"""
        # Get final statistics
        broker_stats = self.broker.get_statistics()
        final_capital = broker_stats['final_balance']
        total_return = broker_stats['total_return']

        # Calculate CAGR (using official test period, not warmup)
        duration_years = (self.end_time - self.official_start_time).days / 365.25
        cagr = (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0

        results = {
            'start_date': self.official_start_time.date().isoformat(),
            'end_date': self.end_time.date().isoformat(),
            'duration_years': duration_years,
            'initial_capital': self.config.capital.initial,
            'final_capital': final_capital,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': self.max_drawdown,
            'total_trades': broker_stats['total_trades'],
            'winning_trades': broker_stats['winning_trades'],
            'losing_trades': broker_stats['losing_trades'],
            'win_rate': broker_stats['win_rate'],
            'open_trades': broker_stats['open_trades'],
            'signals_generated': self.signals_generated,
            'trades_opened': self.trades_opened,
            'trades_closed': self.trades_closed,
            'trades': self.broker.closed_trades,  # All closed trades for export
        }

        # Print results
        if not self.quiet:
            logger.info("="*80)
            logger.info("SIMULATION RESULTS")
            logger.info("="*80)
            logger.info(f"Period: {results['start_date']} to {results['end_date']} ({duration_years:.1f} years)")
            logger.info(f"Initial Capital: ${results['initial_capital']:.2f}")
            logger.info(f"Final Capital:   ${results['final_capital']:.2f}")
            logger.info(f"Total Return:    {results['total_return']:.1%}")
            logger.info(f"CAGR:            {results['cagr']:.1%}")
            logger.info(f"Max Drawdown:    {results['max_drawdown']:.2%}")
            logger.info(f"Total Trades:    {results['total_trades']}")
            logger.info(f"Win Rate:        {results['win_rate']:.1%}")
            logger.info(f"Winners:         {results['winning_trades']}")
            logger.info(f"Losers:          {results['losing_trades']}")
            logger.info(f"Signals Generated: {results['signals_generated']}")
            logger.info("="*80)

        return results
