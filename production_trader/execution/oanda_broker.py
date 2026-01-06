"""
OANDA Broker API Wrapper
=========================
Interfaces with OANDA v20 REST API for live trading.

IMPORTANT: Requires 'v20' package (OANDA Python SDK)
Install with: pip install v20
"""
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import v20
    from v20.order import MarketOrderRequest
    OANDA_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: OANDA v20 SDK import failed: {e}")
    print("Run: pip install v20")
    v20 = None
    MarketOrderRequest = None
    OANDA_AVAILABLE = False


logger = logging.getLogger(__name__)


def parse_oanda_time(time_str: str) -> datetime:
    """
    Parse OANDA timestamp, handling nanosecond precision.

    OANDA returns timestamps with nanoseconds (9 digits), but Python's
    datetime.fromisoformat() only supports microseconds (6 digits).

    Args:
        time_str: OANDA time string (e.g., '2026-01-02T21:59:05.276778139+00:00')

    Returns:
        datetime object
    """
    # Remove 'Z' and replace with '+00:00' for UTC
    time_str = time_str.replace('Z', '+00:00')

    # Handle nanoseconds - truncate to microseconds
    # Format: 2026-01-02T21:59:05.276778139+00:00
    if '.' in time_str and '+' in time_str:
        date_part, tz_part = time_str.split('+')
        if '.' in date_part:
            main_part, frac_part = date_part.split('.')
            # Truncate fractional seconds to 6 digits (microseconds)
            frac_part = frac_part[:6]
            time_str = f"{main_part}.{frac_part}+{tz_part}"

    return datetime.fromisoformat(time_str)


@dataclass
class PriceData:
    """Price data for a currency pair"""
    pair: str
    time: datetime
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float
    mid_close: float


@dataclass
class AccountInfo:
    """Account information"""
    balance: float
    unrealized_pl: float
    margin_used: float
    margin_available: float
    open_trade_count: int


class OandaBroker:
    """
    OANDA broker API wrapper.

    Handles:
    - Real-time price fetching
    - Historical candle data
    - Market order execution
    - Position management
    - Account information
    """

    def __init__(self, api_key: str, account_id: str, account_type: str = 'practice'):
        """
        Initialize OANDA broker connection.

        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            account_type: 'practice' or 'live'
        """
        if not OANDA_AVAILABLE:
            raise ImportError("OANDA v20 SDK not installed. Run: pip install v20")

        self.api_key = api_key
        self.account_id = account_id
        self.account_type = account_type

        # Set API endpoint
        if account_type == 'practice':
            self.hostname = 'api-fxpractice.oanda.com'
        elif account_type == 'live':
            self.hostname = 'api-fxtrade.oanda.com'
        else:
            raise ValueError(f"Invalid account_type: {account_type}")

        # Initialize API context
        self.api = v20.Context(
            hostname=self.hostname,
            token=self.api_key,
            poll_timeout=10.0
        )

        logger.info(f"Initialized OANDA broker ({account_type})")

    def check_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.api.account.get(self.account_id)
            if response.status == 200:
                logger.info("OANDA connection successful")
                return True
            else:
                logger.error(f"OANDA connection failed: {response}")
                return False
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False

    def get_account_summary(self) -> Optional[AccountInfo]:
        """
        Get account summary information.

        Returns:
            AccountInfo object or None if error
        """
        try:
            response = self.api.account.summary(self.account_id)

            if response.status == 200:
                account = response.body['account']
                return AccountInfo(
                    balance=float(account.balance),
                    unrealized_pl=float(account.unrealizedPL),
                    margin_used=float(account.marginUsed),
                    margin_available=float(account.marginAvailable),
                    open_trade_count=int(account.openTradeCount)
                )
            else:
                logger.error(f"Failed to get account summary: {response}")
                return None

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None

    def get_current_prices(self, pairs: List[str]) -> Dict[str, PriceData]:
        """
        Get current bid/ask prices for multiple pairs.

        Args:
            pairs: List of currency pairs (e.g., ['EUR_USD', 'GBP_USD'])

        Returns:
            Dictionary mapping pair to PriceData
        """
        prices = {}

        # Convert pairs to OANDA format (EURUSD -> EUR_USD)
        oanda_pairs = [self._to_oanda_pair(pair) for pair in pairs]

        try:
            response = self.api.pricing.get(
                self.account_id,
                instruments=','.join(oanda_pairs)
            )

            if response.status == 200:
                for price in response.body['prices']:
                    instrument = price.instrument.replace('_', '')
                    time = parse_oanda_time(price.time)

                    bid = float(price.bids[0].price) if price.bids else 0
                    ask = float(price.asks[0].price) if price.asks else 0
                    mid = (bid + ask) / 2

                    prices[instrument] = PriceData(
                        pair=instrument,
                        time=time,
                        bid_open=bid,  # Snapshot, so open = close
                        bid_high=bid,
                        bid_low=bid,
                        bid_close=bid,
                        ask_open=ask,
                        ask_high=ask,
                        ask_low=ask,
                        ask_close=ask,
                        mid_close=mid
                    )

            else:
                logger.error(f"Failed to get prices: {response}")

        except Exception as e:
            logger.error(f"Error getting prices: {e}")

        return prices

    def get_historical_candles(
        self,
        pair: str,
        timeframe: str = 'M15',
        count: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Get historical candle data.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe ('M1', 'M5', 'M15', 'H1', 'H4', 'D')
            count: Number of candles to fetch (max 5000)

        Returns:
            DataFrame with OHLC data or None if error
        """
        oanda_pair = self._to_oanda_pair(pair)

        try:
            response = self.api.instrument.candles(
                instrument=oanda_pair,
                granularity=timeframe,
                count=count,
                price='MBA'  # Mid, Bid, Ask
            )

            if response.status == 200:
                candles = response.body['candles']

                data = []
                for candle in candles:
                    if not candle.complete:
                        continue  # Skip incomplete candles

                    # Calculate spread
                    bid_close = float(candle.bid.c)
                    ask_close = float(candle.ask.c)
                    mid_close = float(candle.mid.c)
                    spread_pct = (ask_close - bid_close) / mid_close

                    data.append({
                        'date': parse_oanda_time(candle.time),
                        # Mid prices (for feature calculation)
                        'open': float(candle.mid.o),
                        'high': float(candle.mid.h),
                        'low': float(candle.mid.l),
                        'close': mid_close,
                        # Bid prices (for short entries and long exits)
                        'bid_open': float(candle.bid.o),
                        'bid_high': float(candle.bid.h),
                        'bid_low': float(candle.bid.l),
                        'bid_close': bid_close,
                        # Ask prices (for long entries and short exits)
                        'ask_open': float(candle.ask.o),
                        'ask_high': float(candle.ask.h),
                        'ask_low': float(candle.ask.l),
                        'ask_close': ask_close,
                        # Spread
                        'spread_pct': spread_pct,
                        'volume': int(candle.volume)
                    })

                df = pd.DataFrame(data)
                df = df.set_index('date')
                return df

            else:
                logger.error(f"Failed to get candles: {response}")
                return None

        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return None

    def place_market_order(
        self,
        pair: str,
        direction: str,
        units: int
    ) -> Optional[str]:
        """
        Place a market order.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            direction: 'long' or 'short'
            units: Number of units (positive for long, negative for short)

        Returns:
            Trade ID if successful, None otherwise
        """
        oanda_pair = self._to_oanda_pair(pair)

        # Ensure units has correct sign
        if direction == 'short' and units > 0:
            units = -units
        elif direction == 'long' and units < 0:
            units = abs(units)

        try:
            # Create market order request
            order_spec = {
                "instrument": oanda_pair,
                "units": str(int(units))  # OANDA expects string
            }

            response = self.api.order.market(
                self.account_id,
                **order_spec
            )

            if response.status in [200, 201]:
                # Try to extract trade ID from response (v20 response.body can be dict-like or object-like)
                try:
                    # First try dict-style access for orderFillTransaction
                    if isinstance(response.body, dict) and 'orderFillTransaction' in response.body:
                        fill_txn = response.body['orderFillTransaction']
                        trade_id = fill_txn.id
                        logger.info(f"Order filled: {pair} {direction} {units} units | Trade ID: {trade_id}")
                        return str(trade_id)
                    # Try attribute-style access
                    elif hasattr(response.body, 'orderFillTransaction'):
                        fill_txn = response.body.orderFillTransaction
                        trade_id = fill_txn.id
                        logger.info(f"Order filled: {pair} {direction} {units} units | Trade ID: {trade_id}")
                        return str(trade_id)
                except (AttributeError, TypeError, KeyError) as e:
                    logger.debug(f"Failed to extract from orderFillTransaction: {e}")

                try:
                    # Fallback: check orderCreateTransaction (dict-style)
                    if isinstance(response.body, dict) and 'orderCreateTransaction' in response.body:
                        create_txn = response.body['orderCreateTransaction']
                        trade_id = create_txn.id
                        logger.info(f"Order created: {pair} {direction} {units} units | Transaction ID: {trade_id}")
                        return str(trade_id)
                    # Try attribute-style access
                    elif hasattr(response.body, 'orderCreateTransaction'):
                        create_txn = response.body.orderCreateTransaction
                        trade_id = create_txn.id
                        logger.info(f"Order created: {pair} {direction} {units} units | Transaction ID: {trade_id}")
                        return str(trade_id)
                except (AttributeError, TypeError, KeyError) as e:
                    logger.debug(f"Failed to extract from orderCreateTransaction: {e}")

                # If we get here, couldn't extract trade ID
                logger.warning(f"Order placed but could not extract trade ID | Response: {response.body}")
                return None
            else:
                # Parse error details from response
                error_msg = str(response)
                if hasattr(response, 'body') and hasattr(response.body, 'errorMessage'):
                    error_msg = response.body.errorMessage
                elif hasattr(response, 'body'):
                    error_msg = f"{response} | Body: {response.body}"

                logger.error(f"Failed to place order: {pair} {direction} {units} units | Error: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def close_trade_by_id(self, trade_id: str) -> bool:
        """
        Close a specific trade by ID.

        Args:
            trade_id: OANDA trade ID

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.api.trade.close(self.account_id, trade_id)

            if response.status in [200, 201, 204]:
                logger.info(f"Closed trade ID: {trade_id}")
                return True
            else:
                logger.error(f"Failed to close trade {trade_id}: {response}")
                return False

        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False

    def close_position(
        self,
        pair: str,
        units: Optional[int] = None,
        trade_id: Optional[str] = None
    ) -> bool:
        """
        Close a position (or part of it).

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            units: Number of units to close (None = close all)
            trade_id: Specific trade ID to close (for netting accounts)

        Returns:
            True if successful, False otherwise
        """
        # If trade_id provided, close that specific trade
        if trade_id:
            return self.close_trade_by_id(trade_id)

        oanda_pair = self._to_oanda_pair(pair)

        try:
            if units is not None:
                # Close specific number of units via opposite market order
                position = self.get_position(pair)
                if position is None:
                    logger.warning(f"No position found for {pair}")
                    return False

                direction = 'short' if position['units'] > 0 else 'long'
                success = self.place_market_order(pair, direction, abs(units)) is not None
                if success:
                    logger.info(f"Closed {abs(units)} units of {pair}")
                return success

            # Close all positions for this pair
            position = self.get_position(pair)
            if not position:
                logger.warning(f"No position found for {pair}")
                return False

            # Determine which side to close based on units (positive = long, negative = short)
            position_units = position['units']

            if position_units > 0:
                # Close long position
                response = self.api.position.close(
                    self.account_id,
                    oanda_pair,
                    longUnits="ALL"
                )
            else:
                # Close short position
                response = self.api.position.close(
                    self.account_id,
                    oanda_pair,
                    shortUnits="ALL"
                )

            if response.status in [200, 201, 204]:
                logger.info(f"Closed position: {pair}")
                return True
            else:
                logger.error(f"Failed to close position: {response}")
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def get_position(self, pair: str) -> Optional[Dict]:
        """
        Get current position for a pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD')

        Returns:
            Dictionary with position info or None if no position
        """
        oanda_pair = self._to_oanda_pair(pair)

        try:
            response = self.api.position.get(
                self.account_id,
                instrument=oanda_pair
            )

            if response.status == 200:
                position = response.body['position']
                units = float(position.long.units) + float(position.short.units)

                if abs(units) < 1:
                    return None  # No position

                return {
                    'pair': pair,
                    'units': units,
                    'avg_price': float(position.long.averagePrice) if units > 0 else float(position.short.averagePrice),
                    'unrealized_pl': float(position.unrealizedPL)
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None

    def get_all_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        positions = []

        try:
            response = self.api.position.list_open(self.account_id)

            if response.status == 200:
                for pos in response.body['positions']:
                    units = float(pos.long.units) + float(pos.short.units)

                    if abs(units) >= 1:
                        positions.append({
                            'pair': pos.instrument.replace('_', ''),
                            'units': units,
                            'avg_price': float(pos.long.averagePrice) if units > 0 else float(pos.short.averagePrice),
                            'unrealized_pl': float(pos.unrealizedPL)
                        })

            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_open_trade_ids(self) -> List[str]:
        """
        Get all open trade IDs.

        Returns:
            List of trade IDs (as strings)
        """
        trade_ids = []

        try:
            response = self.api.trade.list_open(self.account_id)

            if response.status == 200:
                for trade in response.body['trades']:
                    trade_ids.append(trade.id)

            return trade_ids

        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []

    def handle_api_error(self, error: Exception, max_retries: int = 3) -> bool:
        """
        Handle API errors with retry logic.

        Args:
            error: Exception that occurred
            max_retries: Maximum number of retries

        Returns:
            True if recovered, False otherwise
        """
        for attempt in range(max_retries):
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"API error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

            if self.check_connection():
                logger.info("Reconnected successfully")
                return True

        logger.error("Failed to recover from API error")
        return False

    @staticmethod
    def _to_oanda_pair(pair: str) -> str:
        """
        Convert pair format to OANDA format.

        Args:
            pair: Pair like 'EURUSD'

        Returns:
            OANDA format like 'EUR_USD'
        """
        return f"{pair[:3]}_{pair[3:]}"


# Test/demo code
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Try practice account first, fall back to live
    api_key = os.getenv('OANDA_PRACTICE_API_KEY') or os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_PRACTICE_ACCOUNT_ID') or os.getenv('OANDA_ACCOUNT_ID')
    account_type = 'practice' if os.getenv('OANDA_PRACTICE_API_KEY') else 'live'

    if not api_key or not account_id:
        print("ERROR: Please set OANDA credentials in .env file")
        print("For practice: OANDA_PRACTICE_API_KEY and OANDA_PRACTICE_ACCOUNT_ID")
        print("For live: OANDA_API_KEY and OANDA_ACCOUNT_ID")
        exit(1)

    print(f"Testing {account_type.upper()} account...")
    broker = OandaBroker(api_key, account_id, account_type)

    print("Testing OANDA connection...")
    if broker.check_connection():
        print("✓ Connection successful")

        print("\nGetting account summary...")
        account = broker.get_account_summary()
        if account:
            print(f"  Balance: ${account.balance:.2f}")
            print(f"  Open trades: {account.open_trade_count}")

        print("\nGetting current prices...")
        prices = broker.get_current_prices(['EURUSD', 'GBPUSD'])
        for pair, price in prices.items():
            print(f"  {pair}: Bid={price.bid_close:.5f}, Ask={price.ask_close:.5f}")

        print("\nGetting historical candles (EUR_USD 15M)...")
        candles = broker.get_historical_candles('EURUSD', 'M15', count=10)
        if candles is not None:
            print(f"  Retrieved {len(candles)} candles")
            print(candles.tail())

    else:
        print("✗ Connection failed")
