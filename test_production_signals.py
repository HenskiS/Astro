"""
Test Production Signal Generation
==================================
Verify that the production trader can generate signals correctly.
"""
import sys
from pathlib import Path

# Add production_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from production_trader.config import load_config
from production_trader.execution.oanda_broker import OandaBroker
from production_trader.strategies.strategy_15m import Strategy15m

print("=" * 80)
print("TESTING PRODUCTION SIGNAL GENERATION")
print("=" * 80)
print()

# Load config
print("Loading configuration...")
config = load_config('production_trader/config.yaml')
print(f"[OK] Config loaded: {config.oanda.account_type} account")
print(f"[OK] Min confidence: {config.strategy_15m.min_confidence}")
print(f"[OK] Position size: {config.strategy_15m.position_size_pct * 100}%")
print()

# Initialize broker
print("Connecting to OANDA...")
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
if account:
    print(f"[OK] Account balance: ${account.balance:.2f}")
    current_capital = account.balance
else:
    print("[ERROR] Failed to get account balance")
    sys.exit(1)
print()

# Initialize strategy
print("Loading strategy and models...")
try:
    strategy = Strategy15m(config.strategy_15m, broker)
    print(f"[OK] Strategy initialized with {len(strategy.models)} models")
except Exception as e:
    print(f"[ERROR] Failed to initialize strategy: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test signal generation
print("=" * 80)
print("GENERATING SIGNALS")
print("=" * 80)
print()

try:
    signals = strategy.generate_signals(
        current_capital=current_capital,
        existing_positions={}
    )

    print(f"[OK] Signal generation completed successfully")
    print(f"[OK] Generated {len(signals)} signal(s)")
    print()

    if len(signals) > 0:
        print("SIGNALS FOUND:")
        print("-" * 80)
        for i, sig in enumerate(signals, 1):
            print(f"\nSignal {i}:")
            print(f"  Pair:       {sig['pair']}")
            print(f"  Direction:  {sig['direction']}")
            print(f"  Confidence: {sig['confidence']:.1%}")
            print(f"  Size:       {sig['size']:.0f} units")
            print(f"  Target:     {sig['target']:.5f}")
    else:
        print("No signals generated (confidence < 80% or avoid hours)")

except Exception as e:
    print(f"[ERROR] Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("TEST COMPLETE - ALL SYSTEMS OPERATIONAL")
print("=" * 80)
print()
print("[OK] OANDA connection: OK")
print("[OK] Model loading: OK")
print("[OK] Feature calculation: OK")
print("[OK] Signal generation: OK")
print()
print("Ready to run: python production_trader/main.py")
