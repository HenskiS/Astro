"""
Position Manager
================
Tracks and manages all open positions with full state.

Responsibilities:
- Track all open positions
- Update positions with current prices
- Check for exits (targets, stops, ladders, trailing)
- Execute exits via broker
- Log all trades
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from production_trader.execution.oanda_broker import OandaBroker
from production_trader.state.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position"""
    pair: str
    oanda_trade_id: str
    entry_date: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    original_size: float
    breakout_target: float
    confidence: float
    periods_held: int = 0
    max_profit: float = 0.0
    trailing_stop: Optional[float] = None
    peak_price: float = 0.0  # Track peak for trailing stop calculation
    trailing_active: bool = False  # Track if trailing stop is active


class PositionManager:
    """
    Manages all open trading positions.

    Tracks position state, checks exit conditions, and executes
    exits via the broker API.
    """

    def __init__(self, config, broker: OandaBroker, state_manager: StateManager):
        """
        Initialize position manager.

        Args:
            config: Strategy configuration
            broker: OANDA broker instance
            state_manager: State manager instance
        """
        self.config = config
        self.broker = broker
        self.state_manager = state_manager
        self.positions: List[Position] = []

        # Exit parameters from config
        self.emergency_stop_periods = config.emergency_stop_periods
        self.emergency_stop_loss_pct = config.emergency_stop_loss_pct
        self.trailing_stop_trigger = config.trailing_stop_trigger
        self.trailing_stop_pct = config.trailing_stop_pct

        logger.info("PositionManager initialized")

    def open_position(self, signal: Dict) -> bool:
        """
        Open a new position from a signal.

        Args:
            signal: Signal dictionary from strategy

        Returns:
            True if position opened successfully
        """
        try:
            # Place market order
            trade_id = self.broker.place_market_order(
                pair=signal['pair'],
                direction=signal['direction'],
                units=signal['size']
            )

            if trade_id is None:
                logger.error(f"Failed to place order: {signal['pair']} {signal['direction']}")
                return False

            # Get actual entry price from OANDA
            # (In production, OANDA returns the fill price)
            # For now, use current price as approximation
            prices = self.broker.get_current_prices([signal['pair']])
            if not prices:
                logger.error(f"Failed to get entry price for {signal['pair']}")
                return False

            price_data = prices[signal['pair']]
            if signal['direction'] == 'long':
                entry_price = price_data.ask_close
            else:
                entry_price = price_data.bid_close

            # Create position object
            position = Position(
                pair=signal['pair'],
                oanda_trade_id=trade_id,
                entry_date=datetime.now(),
                entry_price=entry_price,
                direction=signal['direction'],
                size=signal['size'],
                original_size=signal['size'],
                breakout_target=signal['target'],
                confidence=signal['confidence']
            )

            self.positions.append(position)

            # Save to state
            self.state_manager.add_position(trade_id, {
                'pair': position.pair,
                'direction': position.direction,
                'entry_price': entry_price,
                'entry_date': position.entry_date.isoformat(),
                'size': position.size,
                'confidence': position.confidence
            })

            logger.info(f"Position opened: {signal['pair']} {signal['direction'].upper()} | "
                       f"Size: {signal['size']} | Entry: {entry_price:.5f} | Trade ID: {trade_id}")

            return True

        except Exception as e:
            logger.error(f"Error opening position: {e}", exc_info=True)
            return False

    def update_all_positions(self) -> int:
        """
        Update all open positions and check for exits.

        Returns:
            Number of positions closed
        """
        if not self.positions:
            return 0

        closed_count = 0

        # Get current prices for all pairs
        pairs = list(set(p.pair for p in self.positions))
        prices = self.broker.get_current_prices(pairs)

        if not prices:
            logger.warning("Failed to get current prices")
            return 0

        # Update each position
        positions_to_remove = []

        for position in self.positions:
            if position.pair not in prices:
                logger.warning(f"No price data for {position.pair}")
                continue

            price_data = prices[position.pair]

            # Update position and check for exit
            exit_info = self._update_position(position, price_data)

            if exit_info:
                # Position should be closed
                reason, exit_price = exit_info

                if self._close_position(position, reason, exit_price):
                    positions_to_remove.append(position)
                    closed_count += 1

        # Remove closed positions
        for position in positions_to_remove:
            self.positions.remove(position)

        if closed_count > 0:
            logger.info(f"Closed {closed_count} positions")

        return closed_count

    def _update_position(
        self,
        position: Position,
        price_data
    ) -> Optional[Tuple[str, float]]:
        """
        Update a position and check for exit conditions.

        Args:
            position: Position to update
            price_data: Current price data

        Returns:
            Tuple of (exit_reason, exit_price) if should exit, None otherwise
        """
        position.periods_held += 1

        # Calculate profits using bid/ask prices
        if position.direction == 'long':
            # Long: exit at BID prices
            current_profit = (price_data.bid_close - position.entry_price) / position.entry_price
            intraday_high_profit = (price_data.bid_high - position.entry_price) / position.entry_price
            hit_target = price_data.bid_high >= position.breakout_target
            current_exit_price = price_data.bid_close
        else:
            # Short: exit at ASK prices
            current_profit = (position.entry_price - price_data.ask_close) / position.entry_price
            intraday_high_profit = (position.entry_price - price_data.ask_low) / position.entry_price
            hit_target = price_data.ask_low <= position.breakout_target
            current_exit_price = price_data.ask_close

        # Update max profit
        position.max_profit = max(position.max_profit, intraday_high_profit)

        # Check immediate stop loss (-5% anytime)
        if hasattr(self.config, 'immediate_stop_loss_pct'):
            if current_profit <= self.config.immediate_stop_loss_pct:
                logger.warning(f"Immediate stop triggered: {position.pair} | "
                             f"P/L: {current_profit:.2%}")
                return ('immediate_stop', current_exit_price)

        # Check emergency stop (24 bars + losing position)
        if position.periods_held >= self.emergency_stop_periods:
            if current_profit < self.emergency_stop_loss_pct:
                logger.warning(f"Emergency stop triggered: {position.pair} | "
                             f"Held: {position.periods_held} periods | P/L: {current_profit:.2%}")
                return ('emergency_stop', current_exit_price)

        # Check trailing stop (activates when target is hit)
        if not position.trailing_active:
            # Only activate trailing stop when target is hit
            if self.trailing_stop_trigger == 'on_target':
                # Check if target was hit this bar
                if hit_target:
                    # Initialize stop at target level
                    position.trailing_stop = position.breakout_target
                    position.trailing_active = True
                    if position.direction == 'long':
                        position.peak_price = price_data.bid_high
                    else:
                        position.peak_price = price_data.ask_low
                    logger.info(f"Trailing stop activated on target hit: {position.pair} | "
                              f"Target: {position.breakout_target:.5f}")
        else:
            # Trailing stop is active - update and check
            if position.direction == 'long':
                # Update peak price
                if price_data.bid_high > position.peak_price:
                    position.peak_price = price_data.bid_high

                # Trail at 75% from TARGET to PEAK (not entry to peak)
                new_stop = position.breakout_target + self.trailing_stop_pct * (position.peak_price - position.breakout_target)
                position.trailing_stop = max(position.trailing_stop, new_stop)

                # Check if stop hit
                if price_data.bid_low <= position.trailing_stop:
                    logger.info(f"Trailing stop hit: {position.pair} | "
                              f"Stop: {position.trailing_stop:.5f} | P/L: {current_profit:.2%}")
                    return ('trailing_stop', position.trailing_stop)

            else:  # short
                # Update peak price
                if price_data.ask_low < position.peak_price:
                    position.peak_price = price_data.ask_low

                # Trail at 75% from TARGET to PEAK
                new_stop = position.breakout_target - self.trailing_stop_pct * (position.breakout_target - position.peak_price)
                position.trailing_stop = min(position.trailing_stop, new_stop)

                # Check if stop hit
                if price_data.ask_high >= position.trailing_stop:
                    logger.info(f"Trailing stop hit: {position.pair} | "
                              f"Stop: {position.trailing_stop:.5f} | P/L: {current_profit:.2%}")
                    return ('trailing_stop', position.trailing_stop)

        # Check target
        if hit_target:
            logger.info(f"Target hit: {position.pair} | P/L: {current_profit:.2%}")
            return ('target', position.breakout_target)

        return None

    def _close_position(self, position: Position, reason: str, exit_price: float) -> bool:
        """
        Close a position completely.

        Args:
            position: Position to close
            reason: Exit reason
            exit_price: Exit price

        Returns:
            True if closed successfully
        """
        try:
            # Close position via OANDA
            success = self.broker.close_position(pair=position.pair)

            if not success:
                logger.error(f"Failed to close position: {position.pair}")
                return False

            # Calculate profit
            if position.direction == 'long':
                raw_profit = (exit_price - position.entry_price) / position.entry_price
            else:
                raw_profit = (position.entry_price - exit_price) / position.entry_price

            # Calculate blended profit (accounting for partial exits)
            profit_pct = self._calculate_blended_profit(position, raw_profit)
            profit_dollars = profit_pct * (position.original_size * position.entry_price)

            # Update capital
            current_capital = self.state_manager.get_capital()
            new_capital = current_capital + profit_dollars
            self.state_manager.set_capital(new_capital)
            self.state_manager.update_daily_pnl(profit_dollars)

            # Remove from state
            self.state_manager.remove_position(position.oanda_trade_id)

            # Log trade
            logger.info(f"Position closed: {position.pair} {position.direction.upper()} | "
                       f"Reason: {reason} | P/L: {profit_pct:.2%} (${profit_dollars:.2f}) | "
                       f"Held: {position.periods_held} periods | Capital: ${new_capital:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def _calculate_blended_profit(self, position: Position, final_profit: float) -> float:
        """Calculate blended profit accounting for partial exits"""
        if not position.partial_exits:
            return final_profit

        # Calculate weighted average
        total_weight = 0
        weighted_profit = 0

        for level, scale_pct in position.partial_exits:
            weighted_profit += level * scale_pct
            total_weight += scale_pct

        # Add remaining position
        remaining_weight = 1 - total_weight
        weighted_profit += final_profit * remaining_weight

        return weighted_profit

    def get_open_positions_by_pair(self) -> Dict[str, List[Position]]:
        """
        Get open positions grouped by pair.

        Returns:
            Dictionary mapping pair to list of positions
        """
        positions_by_pair = {}

        for position in self.positions:
            if position.pair not in positions_by_pair:
                positions_by_pair[position.pair] = []
            positions_by_pair[position.pair].append(position)

        return positions_by_pair

    def get_position_count(self) -> int:
        """Get total number of open positions"""
        return len(self.positions)

    def close_all_positions(self, reason: str = 'manual_close'):
        """
        Close all open positions.

        Args:
            reason: Reason for closing all positions
        """
        logger.warning(f"Closing all {len(self.positions)} positions - Reason: {reason}")

        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            prices = self.broker.get_current_prices([position.pair])

            if prices:
                price_data = prices[position.pair]
                exit_price = price_data.bid_close if position.direction == 'long' else price_data.ask_close
                self._close_position(position, reason, exit_price)

        self.positions.clear()
        logger.info("All positions closed")
