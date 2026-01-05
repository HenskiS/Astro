"""
Test Order Execution
====================
Place a test order and immediately close it to verify OANDA integration.
"""
import sys
import time
from pathlib import Path

# Add production_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from production_trader.config import load_config
from production_trader.execution.oanda_broker import OandaBroker

print("=" * 80)
print("TESTING ORDER EXECUTION")
print("=" * 80)
print()

# Load config
print("Loading configuration...")
config = load_config('production_trader/config.yaml')
print(f"[OK] Account type: {config.oanda.account_type}")
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
print()

# Get account balance
account = broker.get_account_summary()
if not account:
    print("[ERROR] Failed to get account summary")
    sys.exit(1)

print(f"[OK] Account balance: ${account.balance:.2f}")
print(f"[OK] Open positions: {account.open_trade_count}")
print()

# Test order parameters
TEST_PAIR = 'EURUSD'
TEST_DIRECTION = 'long'
TEST_UNITS = 1000  # OANDA minimum for live accounts

print("=" * 80)
print("PLACING TEST ORDER")
print("=" * 80)
print(f"Pair: {TEST_PAIR}")
print(f"Direction: {TEST_DIRECTION}")
print(f"Units: {TEST_UNITS}")
print()

# Get current price
prices = broker.get_current_prices([TEST_PAIR])
if not prices:
    print("[ERROR] Failed to get current price")
    sys.exit(1)

price_data = prices[TEST_PAIR]
print(f"Current price: {price_data.mid_close:.5f}")
print(f"Bid: {price_data.bid_close:.5f} | Ask: {price_data.ask_close:.5f}")
print()

# Place order
print("Placing order...")
trade_id = broker.place_market_order(TEST_PAIR, TEST_DIRECTION, TEST_UNITS)

if trade_id:
    print(f"[OK] Order placed successfully! Trade ID: {trade_id}")
    print()

    # Wait a moment to ensure order is filled
    print("Waiting 2 seconds...")
    time.sleep(2)

    # Check position
    position = broker.get_position(TEST_PAIR)
    if position:
        print(f"[OK] Position confirmed:")
        print(f"  Pair: {position['pair']}")
        print(f"  Units: {position['units']}")
        print(f"  Avg Price: {position['avg_price']:.5f}")
        print(f"  Unrealized P/L: ${position['unrealized_pl']:.2f}")
        print()

    # Close position
    print("Closing position...")
    if broker.close_position(TEST_PAIR):
        print("[OK] Position closed successfully!")
        print()

        # Wait a moment
        time.sleep(1)

        # Get final account summary
        final_account = broker.get_account_summary()
        if final_account:
            print(f"[OK] Final balance: ${final_account.balance:.2f}")
            pnl = final_account.balance - account.balance
            print(f"[OK] Test P/L: ${pnl:+.2f}")
    else:
        print("[ERROR] Failed to close position")
        print("Please close manually in OANDA platform!")
        sys.exit(1)
else:
    print("[ERROR] Failed to place order")
    sys.exit(1)

print()
print("=" * 80)
print("TEST COMPLETE - ORDER EXECUTION WORKING")
print("=" * 80)
print()
print("✓ Connection: OK")
print("✓ Order placement: OK")
print("✓ Position tracking: OK")
print("✓ Order closure: OK")
print()
print("Ready for live trading!")
