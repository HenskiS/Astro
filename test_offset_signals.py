"""
Test Offset Alignment Signals
==============================
Fetch 1m candles, resample to multiple 15m alignments,
and test if we get more signals.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from production_trader.config import load_config
from production_trader.execution.oanda_broker import OandaBroker

print("=" * 80)
print("TESTING OFFSET ALIGNMENT SIGNAL GENERATION")
print("=" * 80)
print()

# Load config
config = load_config('production_trader/config.yaml')
broker = OandaBroker(
    api_key=config.oanda.api_key,
    account_id=config.oanda.account_id,
    account_type=config.oanda.account_type
)

if not broker.check_connection():
    print("[ERROR] Failed to connect")
    sys.exit(1)

print("[OK] Connected to OANDA")

# Test with EURUSD
test_pair = 'EURUSD'
print(f"\nFetching 1-minute candles for {test_pair}...")

# Fetch enough 1m candles to create multiple 15m alignments
# Need: 100 lookback + 24 forward = 124 candles minimum
# Plus some buffer for different alignments = 200 candles
candles_1m = broker.get_historical_candles(test_pair, 'M1', count=200)

if candles_1m is None:
    print("[ERROR] Failed to fetch 1m candles")
    sys.exit(1)

print(f"[OK] Retrieved {len(candles_1m)} 1-minute candles")
print(f"    Date range: {candles_1m.index[0]} to {candles_1m.index[-1]}")
print()

# Function to resample 1m candles to 15m with offset
def resample_to_15m(df, offset_minutes=0):
    """
    Resample 1m candles to 15m candles with offset.

    Args:
        df: DataFrame with 1m OHLCV data
        offset_minutes: Minutes to offset from :00 (0, 5, 10)

    Returns:
        DataFrame with 15m candles
    """
    # Shift index by offset
    df_shifted = df.copy()
    df_shifted.index = df_shifted.index - pd.Timedelta(minutes=offset_minutes)

    # Resample to 15m
    resampled = df_shifted.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'bid_open': 'first',
        'bid_high': 'max',
        'bid_low': 'min',
        'bid_close': 'last',
        'ask_open': 'first',
        'ask_high': 'max',
        'ask_low': 'min',
        'ask_close': 'last',
        'spread_pct': 'mean',
        'volume': 'sum'
    }).dropna()

    # Shift index back
    resampled.index = resampled.index + pd.Timedelta(minutes=offset_minutes)

    return resampled


print("Creating 3 different 15m alignments from 1m data:")
print()

# Create 3 alignments
alignments = {
    'Standard (:00)': resample_to_15m(candles_1m, 0),
    'Offset +5 (:05)': resample_to_15m(candles_1m, 5),
    'Offset +10 (:10)': resample_to_15m(candles_1m, 10)
}

for name, candles in alignments.items():
    print(f"{name}:")
    print(f"  Candles: {len(candles)}")
    print(f"  Latest timestamp: {candles.index[-1]}")
    print(f"  Latest close: {candles['close'].iloc[-1]:.5f}")
    print()

# Check if candle values are different
print("=" * 80)
print("COMPARING CANDLE DATA")
print("=" * 80)
print()

# Compare latest candle from each alignment
print("Latest candle comparison:")
print()
print(f"{'Alignment':<20} {'Timestamp':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10}")
print("-" * 80)
for name, candles in alignments.items():
    latest = candles.iloc[-1]
    ts = candles.index[-1].strftime('%H:%M')
    print(f"{name:<20} {ts:<20} {latest['open']:<10.5f} {latest['high']:<10.5f} {latest['low']:<10.5f} {latest['close']:<10.5f}")

print()
print("Observation:")
print("  Different alignments show different OHLC values (as expected)")
print("  This means technical indicators will be different")
print("  Therefore, predictions will be different")
print()

# Calculate how much the values differ
std_close = alignments['Standard (:00)']['close'].iloc[-1]
off5_close = alignments['Offset +5 (:05)']['close'].iloc[-1]
off10_close = alignments['Offset +10 (:10)']['close'].iloc[-1]

diff_5 = abs(off5_close - std_close)
diff_10 = abs(off10_close - std_close)

print(f"Close price differences:")
print(f"  Standard vs +5min:  {diff_5:.5f} ({diff_5/std_close*100:.3f}%)")
print(f"  Standard vs +10min: {diff_10:.5f} ({diff_10/std_close*100:.3f}%)")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("✓ Feasibility: YES - We can create offset alignments from 1m data")
print()
print("Next steps to implement:")
print("  1. Modify broker to cache 1m candles (200+ candles)")
print("  2. Add resampling logic to create 3 alignments")
print("  3. Run predictions on all 3 alignments")
print("  4. Deduplicate signals (if EURUSD triggers at :00 and :05, only take :00)")
print("  5. Test in backtest to see if it improves performance")
print()
print("Expected benefits:")
print("  - 3x more signal opportunities (every 5 min instead of every 15 min)")
print("  - Catch breakouts faster (don't wait up to 15 min)")
print()
print("Expected costs:")
print("  - More API calls (fetch 200 1m candles vs 100 15m candles)")
print("  - More computation (resample + 3x predictions)")
print("  - Risk of overtrading if signals are highly correlated")
print()
print("Want to implement this? It would require:")
print("  1. Modify OandaBroker.get_historical_candles() to support offset")
print("  2. Modify Strategy15m.generate_signals() to check multiple alignments")
print("  3. Add signal deduplication logic")
print("  4. Backtest to validate performance improvement")
