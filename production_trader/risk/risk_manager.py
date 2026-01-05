"""
Risk Manager
============
Enforces risk limits and safety controls.

CRITICAL for live trading:
- Max positions limits
- Max drawdown circuit breaker
- Daily loss limits
- Position sizing validation
- Emergency shutdown capabilities
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk limits and safety controls.

    Prevents over-leveraging and catastrophic losses through
    strict position limits, drawdown controls, and daily loss limits.
    """

    def __init__(self, config, state_manager):
        """
        Initialize risk manager.

        Args:
            config: Capital configuration
            state_manager: State manager instance
        """
        self.config = config
        self.state_manager = state_manager

        # Risk limits
        self.max_drawdown = config.max_drawdown
        self.daily_loss_limit = config.daily_loss_limit

        # State flags
        self.safe_mode = False
        self.emergency_shutdown = False

        logger.info(f"RiskManager initialized | Max DD: {self.max_drawdown:.1%} | "
                   f"Daily loss: {self.daily_loss_limit:.1%}")

    def check_max_drawdown(self, current_capital: float) -> bool:
        """
        Check if max drawdown exceeded.

        Args:
            current_capital: Current account capital

        Returns:
            True if max drawdown exceeded (EMERGENCY)
        """
        peak_capital = self.state_manager.get_peak_capital()
        drawdown = (current_capital - peak_capital) / peak_capital

        if drawdown <= -self.max_drawdown:
            logger.critical(f"MAX DRAWDOWN EXCEEDED: {drawdown:.1%} | "
                          f"Peak: ${peak_capital:.2f} | Current: ${current_capital:.2f}")
            self.emergency_shutdown = True
            return True

        # Warning at 75% of max
        warning_threshold = -self.max_drawdown * 0.75
        if drawdown <= warning_threshold:
            logger.warning(f"Drawdown warning: {drawdown:.1%} "
                         f"(threshold: {warning_threshold:.1%})")

        return False

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit exceeded.

        Returns:
            True if daily loss limit exceeded (enter safe mode)
        """
        daily_pnl = self.state_manager.get_daily_pnl()
        current_capital = self.state_manager.get_capital()
        daily_loss_pct = daily_pnl / current_capital

        if daily_loss_pct <= -self.daily_loss_limit:
            logger.critical(f"DAILY LOSS LIMIT EXCEEDED: {daily_loss_pct:.1%} "
                          f"(${daily_pnl:.2f})")
            self.safe_mode = True
            return True

        # Warning at 75% of limit
        warning_threshold = -self.daily_loss_limit * 0.75
        if daily_loss_pct <= warning_threshold:
            logger.warning(f"Daily loss warning: {daily_loss_pct:.1%} "
                         f"(threshold: {warning_threshold:.1%})")

        return False

    def check_position_limits(
        self,
        signal: Dict,
        current_positions: Dict[str, List],
        strategy_config
    ) -> bool:
        """
        Check if position limits allow opening new position.

        Args:
            signal: Signal to check
            current_positions: Current positions by pair
            strategy_config: Strategy configuration

        Returns:
            True if can open position, False if limits exceeded
        """
        # Check total position limit
        total_positions = sum(len(positions) for positions in current_positions.values())
        if total_positions >= strategy_config.max_positions_total:
            logger.debug(f"Total position limit reached: {total_positions}/{strategy_config.max_positions_total}")
            return False

        # Check per-pair limit
        pair_positions = current_positions.get(signal['pair'], [])
        if len(pair_positions) >= strategy_config.max_positions_per_pair:
            logger.debug(f"Pair position limit reached: {signal['pair']} "
                       f"{len(pair_positions)}/{strategy_config.max_positions_per_pair}")
            return False

        return True

    def check_kill_switch(self) -> bool:
        """
        Check for KILL_SWITCH file.

        Returns:
            True if kill switch activated
        """
        kill_switch_file = Path('KILL_SWITCH')

        if kill_switch_file.exists():
            logger.critical("KILL_SWITCH FILE DETECTED - EMERGENCY SHUTDOWN")
            self.emergency_shutdown = True
            return True

        return False

    def validate_position_size(
        self,
        signal: Dict,
        current_capital: float,
        pair_price: float
    ) -> Optional[int]:
        """
        Validate and adjust position size if needed.

        Args:
            signal: Signal with position size
            current_capital: Current account capital
            pair_price: Current price of the pair

        Returns:
            Validated position size, or None if invalid
        """
        requested_size = signal['size']

        # Sanity check: reject if absurdly large (>50% of capital)
        max_reasonable_size = (current_capital * 0.50) / pair_price
        if requested_size > max_reasonable_size:
            logger.error(f"Position size {requested_size} exceeds sanity limit (50% capital = {max_reasonable_size:.0f} units)")
            return None

        # Reject if too small (below minimum)
        if requested_size < 1:
            logger.warning(f"Position size too small: {requested_size}")
            return None

        # Validate OANDA minimum (live accounts typically require 1000 units minimum)
        if requested_size < 1000:
            logger.warning(f"Position size {requested_size} below OANDA minimum (1000 units) | Rounding up to 1000")
            return 1000

        return int(requested_size)

    def can_trade(self) -> bool:
        """
        Check if trading is allowed.

        Returns:
            True if can trade, False if in safe mode or emergency shutdown
        """
        if self.emergency_shutdown:
            logger.critical("Trading DISABLED - Emergency shutdown active")
            return False

        if self.safe_mode:
            logger.warning("Trading DISABLED - Safe mode active (daily loss limit)")
            return False

        return True

    def reset_safe_mode(self):
        """Reset safe mode (call at start of new day)"""
        if self.safe_mode:
            logger.info("Resetting safe mode for new trading day")
            self.safe_mode = False

    def get_status(self) -> Dict:
        """
        Get current risk status.

        Returns:
            Dictionary with risk status info
        """
        current_capital = self.state_manager.get_capital()
        peak_capital = self.state_manager.get_peak_capital()
        daily_pnl = self.state_manager.get_daily_pnl()

        drawdown = (current_capital - peak_capital) / peak_capital
        daily_loss_pct = daily_pnl / current_capital

        return {
            'can_trade': self.can_trade(),
            'safe_mode': self.safe_mode,
            'emergency_shutdown': self.emergency_shutdown,
            'current_capital': current_capital,
            'peak_capital': peak_capital,
            'drawdown': drawdown,
            'drawdown_limit': -self.max_drawdown,
            'daily_pnl': daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'daily_loss_limit': -self.daily_loss_limit
        }


# Create __init__.py for risk package
if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Test risk manager
    print("Testing RiskManager...")

    # Mock config
    class MockConfig:
        max_drawdown = 0.15
        daily_loss_limit = 0.05
        risk_per_trade = 0.004

    class MockState:
        def __init__(self):
            self.capital = 500
            self.peak_capital = 600
            self.daily_pnl = -20

        def get_capital(self):
            return self.capital

        def get_peak_capital(self):
            return self.peak_capital

        def get_daily_pnl(self):
            return self.daily_pnl

    config = MockConfig()
    state = MockState()
    risk_mgr = RiskManager(config, state)

    # Test drawdown check
    print("\n1. Testing drawdown check...")
    exceeded = risk_mgr.check_max_drawdown(500)
    print(f"   Current: $500, Peak: $600, Drawdown: {(500-600)/600:.1%}")
    print(f"   Exceeded: {exceeded}")

    # Test daily loss check
    print("\n2. Testing daily loss check...")
    exceeded = risk_mgr.check_daily_loss_limit()
    print(f"   Daily P/L: ${-20}, Limit: {-0.05:.1%}")
    print(f"   Exceeded: {exceeded}")

    # Test position size validation
    print("\n3. Testing position size validation...")
    signal = {'pair': 'EURUSD', 'size': 5000}
    validated_size = risk_mgr.validate_position_size(signal, 500, 1.10)
    print(f"   Requested: {signal['size']}, Validated: {validated_size}")

    print("\n✓ RiskManager tests complete")
