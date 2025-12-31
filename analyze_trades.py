"""
ANALYZE TRADES
==============
Look at the trade distribution from production simulation
"""
import pandas as pd
import pickle

print("="*100)
print("TRADE ANALYSIS")
print("="*100)
print()

# We need to extract the trades from the last run
# The production simulation should have saved trades in the results

# For now, let's create a quick script to re-extract key stats
# We'll check the production simulation output

print("Key metrics from production simulation:")
print("  Final Capital: -$168,408")
print("  Initial Capital: $500")
print("  Total Loss: $168,908")
print("  Total Trades: 14,812")
print("  Win Rate: 88.2%")
print("  Winning Trades: ~13,062")
print("  Losing Trades: ~1,750")
print()

# Calculate what the average trade sizes must be
winning_trades = 14812 * 0.882
losing_trades = 14812 * 0.118

total_loss = 168908

print("If win rate is 88.2% and we lost $168,908:")
print(f"  Winning trades: {winning_trades:.0f}")
print(f"  Losing trades: {losing_trades:.0f}")
print()

# Let's assume average winner is X
# Then: (winning_trades * X) - (losing_trades * Y) = -168,908
# We need to figure out X and Y

# From quarterly backtest: $500 -> $34.8M = $34,799,500 profit
# That's 69,599x return
# If they had similar trade count, average winning trade in backtest was much larger

print("HYPOTHESIS: The issue might be:")
print("  1. Position sizes are way too large (overleverage)")
print("  2. Stop losses not working properly")
print("  3. Ladder exits not scaling out properly")
print("  4. Capital going negative means margin call / unlimited loss")
print()

print("The fact that capital went NEGATIVE (-$168K) suggests:")
print("  - No proper stop loss enforcement")
print("  - Allowing losses to run unlimited")
print("  - Positions not being closed when capital depleted")
print()
