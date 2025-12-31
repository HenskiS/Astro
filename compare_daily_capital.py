"""
Compare daily capital progression between backtest and production for first few days
to identify exactly where they diverge
"""
import pandas as pd
import pickle

print("="*100)
print("DAILY CAPITAL COMPARISON - FIRST WEEK")
print("="*100)
print()

# This would require running both systems with daily capital logging
# For now, let's just outline what to check:

print("To diagnose the capital divergence, we need to track:")
print()
print("1. CAPITAL PROGRESSION")
print("   - Start: $500")
print("   - After each trade: $X")
print("   - End of day 1, day 2, etc.")
print()
print("2. TRADES EXECUTED")
print("   - Pair, direction, entry price, size")
print("   - Exit price, exit reason")
print("   - Profit/loss in dollars")
print()
print("3. POSITION SIZING")
print("   - Capital at time of entry")
print("   - Risk amount (capital * 0.007)")
print("   - Position size calculation")
print()
print("4. PROFIT CALCULATION")
print("   - Raw profit percentage")
print("   - Blended profit (with ladder exits)")
print("   - Profit in dollars")
print("   - New capital")
print()

print("KEY QUESTION: Do they diverge on Day 1, or later?")
print("- If Day 1: Position sizing or entry logic bug")
print("- If later: Compounding or exit logic bug")
print()
