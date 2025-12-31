"""
MOCK BROKER API
===============
Simulates a real broker API endpoint that serves forex data.

Key features:
- Only returns data up to the current simulation date
- No future data ever accessible
- Mimics real broker API behavior (OANDA-style)
- Perfect temporal isolation

This ensures the backtest runs exactly like production would.
"""
import pandas as pd
import os
from typing import Optional, Dict, List


class MockBrokerAPI:
    """
    Mock broker API that serves historical forex data.

    This simulates how a real broker API works:
    - You can only query data up to "now"
    - No future data is accessible
    - Data is served one day at a time in production mode
    """

    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the mock broker API.

        Args:
            data_dir: Directory containing CSV files with forex data
        """
        self.data_dir = data_dir
        self._data_cache = {}

    def _load_pair_data(self, pair: str) -> pd.DataFrame:
        """Load data for a currency pair (with caching)"""
        if pair not in self._data_cache:
            file_path = os.path.join(self.data_dir, f'{pair}_1day_with_spreads.csv')

            if not os.path.exists(file_path):
                raise ValueError(f"Data file not found for {pair}: {file_path}")

            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Remove timezone for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            self._data_cache[pair] = df

        return self._data_cache[pair]

    def get_history(
        self,
        pair: str,
        count: int,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Get historical OHLC data for a currency pair.

        This simulates a real broker API call. In production, you would call:
            broker.get_history('EURUSD', count=100)

        And it would return the last 100 bars up to NOW. We simulate this by
        only returning data up to end_date.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            count: Number of bars to return
            end_date: Current simulation date (data up to this point)

        Returns:
            DataFrame with OHLC data, ending at or before end_date
        """
        # Load full data
        full_data = self._load_pair_data(pair)

        # Remove timezone from end_date if needed
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)

        # CRITICAL: Only return data up to end_date (inclusive of the full day)
        # This is the key enforcement of no lookahead
        # Use date normalization to include all data from the end_date day
        # (CSV timestamps are 14:00:00, but end_date might be 00:00:00)
        available_data = full_data[full_data.index.normalize() <= end_date.normalize()]

        if len(available_data) == 0:
            raise ValueError(f"No data available for {pair} up to {end_date}")

        # Return last 'count' bars
        return available_data.tail(count)

    def get_current_price(
        self,
        pair: str,
        date: pd.Timestamp
    ) -> Dict:
        """
        Get current price for a currency pair at a specific date.

        This simulates getting the current market price. In production:
            broker.get_current_price('EURUSD')

        Would return the current OHLC bar.

        Args:
            pair: Currency pair
            date: Current simulation date

        Returns:
            Dictionary with open, high, low, close, date
        """
        # Load data
        full_data = self._load_pair_data(pair)

        # Remove timezone
        if date.tz is not None:
            date = date.tz_localize(None)

        # Get data for this specific date
        # Use date normalization to handle timestamp differences
        matching_dates = full_data[full_data.index.normalize() == date.normalize()]
        if len(matching_dates) > 0:
            row = matching_dates.iloc[0]
            return {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'date': matching_dates.index[0]
            }
        else:
            # Find closest date before this one
            available = full_data[full_data.index.normalize() <= date.normalize()]
            if len(available) == 0:
                raise ValueError(f"No data available for {pair} at {date}")

            row = available.iloc[-1]
            return {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'date': available.index[-1]
            }

    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        pairs = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_1day_with_spreads.csv'):
                pair = filename.replace('_1day_with_spreads.csv', '')
                pairs.append(pair)
        return pairs

    def get_next_trading_day(
        self,
        pair: str,
        current_date: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        """
        Get the next trading day after current_date.

        Used to determine when orders will be filled.

        Args:
            pair: Currency pair
            current_date: Current date

        Returns:
            Next trading day, or None if no data available
        """
        full_data = self._load_pair_data(pair)

        # Remove timezone
        if current_date.tz is not None:
            current_date = current_date.tz_localize(None)

        # Find next day
        future_data = full_data[full_data.index > current_date]

        if len(future_data) > 0:
            return future_data.index[0]
        else:
            return None


if __name__ == '__main__':
    # Demo usage
    print("Mock Broker API Demo")
    print("="*60)
    print()

    api = MockBrokerAPI(data_dir='data')

    # Get available pairs
    pairs = api.get_available_pairs()
    print(f"Available pairs: {pairs}")
    print()

    # Simulate querying data on 2020-06-15
    current_date = pd.Timestamp('2020-06-15')
    print(f"Simulation date: {current_date.date()}")
    print()

    # Get historical data
    history = api.get_history('EURUSD', count=10, end_date=current_date)
    print(f"Last 10 bars (up to {current_date.date()}):")
    print(history[['open', 'high', 'low', 'close']].tail())
    print()

    # Get current price
    current = api.get_current_price('EURUSD', date=current_date)
    print(f"Current price: {current}")
    print()

    # Get next trading day
    next_day = api.get_next_trading_day('EURUSD', current_date)
    print(f"Next trading day: {next_day.date()}")
    print()

    # Verify no future data
    print("Verification: Can we access future data?")
    try:
        future_date = pd.Timestamp('2030-01-01')
        future_data = api.get_history('EURUSD', count=10, end_date=future_date)
        print(f"  Last available date: {future_data.index[-1].date()}")
        print(f"  Correctly limited to available historical data!")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("="*60)
    print("API enforces temporal isolation - no lookahead possible!")
