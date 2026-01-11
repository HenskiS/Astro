"""
Mock Broker for Simulation
===========================
Replaces OandaBroker with historical data replay.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from production_trader.execution.oanda_broker import PriceData, AccountInfo

logger = logging.getLogger(__name__)


class MockTrade:
    """Represents a simulated trade"""
    def __init__(self, trade_id: str, pair: str, units: float, entry_price: float,
                 entry_time: datetime, direction: str, position_value_usd: float):
        self.trade_id = trade_id
        self.pair = pair
        self.units = units
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction  # 'long' or 'short'
        self.position_value_usd = position_value_usd  # Dollar value of position

    def get_unrealized_pl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L in USD using percentage returns.

        This matches the backtest approach and avoids currency conversion complexities.

        Args:
            current_price: Current price of the pair

        Returns:
            P&L in USD
        """
        # Calculate percentage return
        if self.direction == 'long':
            pct_return = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pct_return = (self.entry_price - current_price) / self.entry_price

        # Apply percentage to USD position value (matches backtest)
        return pct_return * self.position_value_usd


class MockBroker:
    """
    Mock broker that replays historical data.

    Implements the same interface as OandaBroker but uses historical
    CSV data instead of making real API calls.
    """

    def __init__(self, data_dir: str, pairs: List[str], initial_balance: float,
                 current_time: datetime = None):
        """
        Initialize mock broker.

        Args:
            data_dir: Directory containing historical CSV files
            pairs: List of pairs to load (e.g., ['EURUSD', 'GBPUSD'])
            initial_balance: Starting account balance
            current_time: Current simulation time (None = use first available data)
        """
        self.data_dir = data_dir
        self.pairs = pairs
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_time = current_time

        # Trade tracking
        self.open_trades = {}  # trade_id -> MockTrade
        self.closed_trades = []
        self.next_trade_id = 1

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Load historical data
        self.historical_data = {}
        self._load_data()

        logger.info(f"MockBroker initialized: ${initial_balance:.2f} balance, {len(pairs)} pairs")

    def _load_data(self):
        """Load historical price data for all pairs"""
        for pair in self.pairs:
            file_path = f"{self.data_dir}/{pair}_15m.csv"
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                self.historical_data[pair] = df
                logger.info(f"Loaded {len(df):,} bars for {pair}")
            except Exception as e:
                logger.error(f"Failed to load data for {pair}: {e}")
                self.historical_data[pair] = pd.DataFrame()

    def set_current_time(self, current_time: datetime):
        """Update the current simulation time"""
        self.current_time = current_time

    def check_connection(self) -> bool:
        """Always returns True for simulation"""
        return True

    def get_account_summary(self) -> Optional[AccountInfo]:
        """Return simulated account information"""
        # Calculate unrealized P&L from open positions
        unrealized_pl = 0.0
        for trade in self.open_trades.values():
            # Get current price
            prices = self.get_current_prices([trade.pair])
            if prices and trade.pair in prices:
                price_data = prices[trade.pair]
                # Use bid for longs, ask for shorts
                current_price = price_data.bid_close if trade.direction == 'long' else price_data.ask_close
                unrealized_pl += trade.get_unrealized_pl(current_price)

        return AccountInfo(
            balance=self.balance + unrealized_pl,
            unrealized_pl=unrealized_pl,
            margin_used=0.0,  # Simplified
            margin_available=self.balance,
            open_trade_count=len(self.open_trades)
        )

    def get_current_prices(self, pairs: List[str]) -> Dict[str, PriceData]:
        """
        Get current prices for given pairs at current_time.

        Returns the OHLC data for the bar that contains current_time.
        """
        if self.current_time is None:
            logger.warning("current_time not set")
            return {}

        result = {}
        for pair in pairs:
            if pair not in self.historical_data:
                continue

            df = self.historical_data[pair]

            # Find the most recent bar at or before current_time
            mask = df.index <= self.current_time
            if not mask.any():
                continue

            # Get the most recent bar
            recent_bars = df[mask]
            if len(recent_bars) == 0:
                continue

            bar = recent_bars.iloc[-1]
            bar_time = recent_bars.index[-1]

            result[pair] = PriceData(
                pair=pair,
                time=bar_time,
                bid_open=bar['bid_open'],
                bid_high=bar['bid_high'],
                bid_low=bar['bid_low'],
                bid_close=bar['bid_close'],
                ask_open=bar['ask_open'],
                ask_high=bar['ask_high'],
                ask_low=bar['ask_low'],
                ask_close=bar['ask_close'],
                mid_close=bar['close']
            )

        return result

    def get_historical_candles(self, pair: str, count: int, timeframe: str = 'M15') -> Optional[pd.DataFrame]:
        """
        Get historical candle data ending at current_time.

        Args:
            pair: Currency pair
            count: Number of candles to return
            timeframe: Timeframe (e.g., 'M15' for 15 minutes)

        Returns:
            DataFrame with OHLC data or None if not available
        """
        if self.current_time is None:
            logger.warning("current_time not set")
            return None

        if pair not in self.historical_data:
            logger.warning(f"No historical data for {pair}")
            return None

        df = self.historical_data[pair]

        # Get candles up to and including current_time
        mask = df.index <= self.current_time
        if not mask.any():
            return None

        recent_data = df[mask]

        # Return last 'count' candles
        if len(recent_data) < count:
            return recent_data  # Return all available if less than requested

        return recent_data.tail(count)

    def place_market_order(self, pair: str, direction: str, units: int, position_value_usd: float = None) -> Optional[str]:
        """
        Simulate placing a market order.

        Args:
            pair: Currency pair
            direction: 'long' or 'short'
            units: Number of units (positive for long, negative for short)
            position_value_usd: Dollar value of the position (for P&L calculation)

        Returns:
            Trade ID if successful, None otherwise
        """
        if self.current_time is None:
            logger.error("Cannot place order: current_time not set")
            return None

        # Get current price
        prices = self.get_current_prices([pair])
        if not prices or pair not in prices:
            logger.error(f"Cannot place order: no price data for {pair}")
            return None

        price_data = prices[pair]

        # Determine entry price based on direction (with spread)
        if direction == 'long':
            entry_price = price_data.ask_open  # Buy at ask
        elif direction == 'short':
            entry_price = price_data.bid_open  # Sell at bid
        else:
            logger.error(f"Invalid direction: {direction}")
            return None

        # Calculate position value if not provided
        if position_value_usd is None:
            # Estimate based on units and price (works for most pairs)
            if pair.startswith('USD'):
                # USD-base: 1 unit = $1
                position_value_usd = abs(units)
            else:
                # USD-quote or cross: approximate
                position_value_usd = abs(units) * entry_price

        # Create trade
        trade_id = str(self.next_trade_id)
        self.next_trade_id += 1

        trade = MockTrade(
            trade_id=trade_id,
            pair=pair,
            units=units,
            entry_price=entry_price,
            entry_time=self.current_time,
            direction=direction,
            position_value_usd=position_value_usd
        )

        self.open_trades[trade_id] = trade
        self.total_trades += 1

        logger.info(f"MockOrder placed: {pair} {direction} {abs(units):.0f} units @ {entry_price:.5f}")

        return trade_id

    def close_trade_by_id(self, trade_id: str, exit_price: Optional[float] = None) -> bool:
        """Close a trade by ID with optional specific exit price"""
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found")
            return False

        trade = self.open_trades[trade_id]

        # Use provided exit_price or get current market price
        if exit_price is None:
            prices = self.get_current_prices([trade.pair])
            if not prices or trade.pair not in prices:
                logger.error(f"Cannot close trade: no price data for {trade.pair}")
                return False

            price_data = prices[trade.pair]

            # Determine exit price (with spread)
            if trade.direction == 'long':
                exit_price = price_data.bid_close  # Sell at bid
            else:
                exit_price = price_data.ask_close  # Buy back at ask

        # Calculate P&L
        pl = trade.get_unrealized_pl(exit_price)
        old_balance = self.balance
        self.balance += pl

        # Debug logging for balance updates
        logger.info(f"[BALANCE UPDATE] {trade.pair} closed: ${old_balance:.2f} -> ${self.balance:.2f} (P&L: ${pl:+.2f})")

        # Track statistics
        if pl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Store closed trade
        # Calculate pl_pct from price movement (matches backtest approach)
        if trade.direction == 'long':
            pl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # short
            pl_pct = (trade.entry_price - exit_price) / trade.entry_price

        self.closed_trades.append({
            'trade_id': trade_id,
            'pair': trade.pair,
            'direction': trade.direction,
            'units': trade.units,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'entry_time': trade.entry_time,
            'exit_time': self.current_time,
            'pl': pl,
            'pl_pct': pl_pct
        })

        logger.info(f"MockTrade closed: {trade.pair} {trade.direction} P&L=${pl:.2f}")

        # Remove from open trades
        del self.open_trades[trade_id]

        return True

    def close_position(self, pair: str, units: Optional[float] = None,
                      trade_id: Optional[str] = None, exit_price: Optional[float] = None) -> bool:
        """
        Close a position (or part of it).

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            units: Number of units to close (None = close all)
            trade_id: Specific trade ID to close (for netting accounts)
            exit_price: Specific exit price to use (e.g., for stop losses)

        Returns:
            True if successful, False otherwise
        """
        # If trade_id specified, close that specific trade
        if trade_id is not None:
            return self.close_trade_by_id(trade_id, exit_price=exit_price)

        # Otherwise, find all trades for this pair
        trades_to_close = [tid for tid, t in self.open_trades.items() if t.pair == pair]

        if not trades_to_close:
            logger.warning(f"No open trades for {pair}")
            return False

        # Close all trades for this pair
        success = True
        for tid in trades_to_close:
            if not self.close_trade_by_id(tid):
                success = False

        return success

    def get_position(self, pair: str) -> Optional[Dict]:
        """Get position information for a pair"""
        # Sum up all trades for this pair
        total_units = 0.0
        avg_entry_price = 0.0
        pl = 0.0

        for trade in self.open_trades.values():
            if trade.pair == pair:
                total_units += trade.units

                # Get current price
                prices = self.get_current_prices([pair])
                if prices and pair in prices:
                    price_data = prices[pair]
                    current_price = price_data.bid_close if trade.direction == 'long' else price_data.ask_close
                    pl += trade.get_unrealized_pl(current_price)

        if total_units == 0:
            return None

        return {
            'pair': pair,
            'units': total_units,
            'pl': pl,
            'long_units': total_units if total_units > 0 else 0,
            'short_units': total_units if total_units < 0 else 0
        }

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = []
        seen_pairs = set()

        for trade in self.open_trades.values():
            if trade.pair not in seen_pairs:
                pos = self.get_position(trade.pair)
                if pos:
                    positions.append(pos)
                seen_pairs.add(trade.pair)

        return positions

    def get_open_trade_ids(self) -> List[str]:
        """Get list of open trade IDs"""
        return list(self.open_trades.keys())

    def handle_api_error(self, error: Exception, max_retries: int = 3) -> bool:
        """Mock error handler (always returns True)"""
        return True

    @staticmethod
    def _to_oanda_pair(pair: str) -> str:
        """Convert pair format (mock, just returns same)"""
        return pair

    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'open_trades': len(self.open_trades)
        }
