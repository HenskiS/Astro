"""
Close All EURUSD Positions
===========================
Quick script to close test positions.
"""
import sys
from pathlib import Path

# Add production_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from production_trader.config import load_config
from production_trader.execution.oanda_broker import OandaBroker

print("Connecting to OANDA...")
config = load_config('production_trader/config.yaml')
broker = OandaBroker(
    api_key=config.oanda.api_key,
    account_id=config.oanda.account_id,
    account_type=config.oanda.account_type
)

if not broker.check_connection():
    print("[ERROR] Failed to connect")
    sys.exit(1)

print("[OK] Connected")

# Get current positions
account = broker.get_account_summary()
print(f"\nCurrent open positions: {account.open_trade_count}")

# Close EURUSD
print("\nClosing EURUSD positions...")
if broker.close_position('EURUSD'):
    print("[OK] EURUSD positions closed")
else:
    print("[ERROR] Failed to close EURUSD")

# Check final state
account = broker.get_account_summary()
print(f"\nFinal open positions: {account.open_trade_count}")
print(f"Account balance: ${account.balance:.2f}")
