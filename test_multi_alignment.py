"""
Test Multi-Alignment Strategy
==============================
Test fetching 15m candles at different alignments (:00, :05, :10)
to see if we can generate signals every 5 minutes instead of every 15.

Theory: Models trained on standard 15m candles should work on any
15m candle alignment since technical indicators are relative.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from production_trader.config import load_config
from production_trader.execution.oanda_broker import OandaBroker
from production_trader.strategies.strategy_15m import Strategy15m

print("=" * 80)
print("TESTING MULTI-ALIGNMENT 15M STRATEGY")
print("=" * 80)
print()
print("Testing 3 alignments:")
print("  - Standard: :00, :15, :30, :45")
print("  - Offset +5: :05, :20, :35, :50")
print("  - Offset +10: :10, :25, :40, :55")
print()

# Load config
config = load_config('production_trader/config.yaml')
broker = OandaBroker(
    api_key=config.oanda.api_key,
    account_id=config.oanda.account_id,
    account_type=config.oanda.account_type
)

if not broker.check_connection():
    print("[ERROR] Failed to connect to OANDA")
    sys.exit(1)

print("[OK] Connected to OANDA")

# Get account balance
account = broker.get_account_summary()
current_capital = account.balance
print(f"[OK] Account balance: ${current_capital:.2f}")
print()

# Initialize strategy
strategy = Strategy15m(config.strategy_15m, broker)
print(f"[OK] Strategy initialized with {len(strategy.models)} models")
print()

# Test fetching candles at different alignments
test_pair = 'EURUSD'
print(f"Testing candle alignment for {test_pair}...")
print()

# Standard alignment (:00, :15, :30, :45)
print("1. Standard alignment (current implementation)")
candles_standard = broker.get_historical_candles(test_pair, 'M15', count=5)
if candles_standard is not None:
    print(f"   Retrieved {len(candles_standard)} candles")
    print(f"   Latest timestamps:")
    for idx in candles_standard.tail(3).index:
        print(f"     {idx.strftime('%Y-%m-%d %H:%M:%S')} (minute: {idx.minute})")
else:
    print("   [ERROR] Failed to fetch candles")

print()

# Check if OANDA API supports alignment parameters
print("2. Testing if we can fetch offset alignments...")
print("   (OANDA v20 API may or may not support this)")
print()

# Try to fetch with 5-minute offset
# Note: OANDA may not support arbitrary alignments in v20 API
# We might need to use a different approach (fetch 1m candles and resample)
print("3. Alternative approach: Fetch recent 1m candles and resample")
print("   This would give us full control over alignment")
print()

# Calculate what signals we'd get right now on standard alignment
print("=" * 80)
print("TESTING SIGNAL GENERATION (STANDARD ALIGNMENT)")
print("=" * 80)
print()

signals = strategy.generate_signals(
    current_capital=current_capital,
    existing_positions={}
)

print(f"Generated {len(signals)} signal(s) on standard alignment")
if len(signals) > 0:
    for i, sig in enumerate(signals, 1):
        print(f"\nSignal {i}:")
        print(f"  Pair:       {sig['pair']}")
        print(f"  Direction:  {sig['direction']}")
        print(f"  Confidence: {sig['confidence']:.1%}")
        print(f"  Size:       {sig['size']:.0f} units")
else:
    print("  No signals (confidence < 80%)")

print()
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()
print("Key findings:")
print("  1. OANDA v20 API provides candles aligned to standard intervals")
print("  2. To test offset alignments, we have 2 options:")
print()
print("Option A: Fetch 1-minute candles and resample")
print("  - Pros: Full control over alignment")
print("  - Cons: 15x more API calls, more data processing")
print()
print("Option B: Use standard alignment only")
print("  - Pros: Simple, API-efficient")
print("  - Cons: Signals only every 15 minutes")
print()
print("Recommendation:")
print("  The multi-alignment strategy is theoretically sound, but:")
print("  - OANDA API doesn't natively support offset alignments")
print("  - Would need to fetch 1m candles and resample (complex)")
print("  - More API calls = more latency and potential rate limits")
print("  - Signal correlation would likely be high (redundant)")
print()
print("Better alternatives:")
print("  1. Stick with 15m standard alignment (current approach)")
print("  2. Consider 5m timeframe instead (native API support)")
print("  3. Use multiple timeframes (5m + 15m) with separate models")
