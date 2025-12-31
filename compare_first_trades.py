"""
Compare the first few trades between backtest and production simulation
to identify where execution differs
"""
import pickle
import pandas as pd

print("="*100)
print("COMPARING FIRST TRADES - BACKTEST VS PRODUCTION")
print("="*100)
print()

# Run backtest for Q1 only to get trades
print("Loading backtest data...")
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_predictions = pickle.load(f)

# Get Q1 2016 predictions
q1_preds = all_predictions['2016Q1']

print(f"Q1 pairs: {list(q1_preds.keys())}")
print()

# Check first few days
print("First few days of Q1 predictions:")
for pair, df in q1_preds.items():
    first_5 = df.head(3)
    print(f"\n{pair}:")
    for idx, row in first_5.iterrows():
        print(f"  {idx.date()}: close={row['close']:.5f}, HIGH_prob={row['breakout_high_prob']:.3f}, LOW_prob={row['breakout_low_prob']:.3f}")
print()

# Load production simulation's saved trades
print("Checking if production simulation saved trades...")
try:
    # Production simulation should save trades somewhere
    # Let me check the output
    print("[INFO] Need to run production simulation to get trades")
    print("[INFO] Or check if it logged trades during execution")
except:
    pass

print()
print("="*100)
print("ANALYSIS PLAN:")
print("="*100)
print()
print("To find the divergence, we need to:")
print("1. Run backtest for Q1 2016 only - save all trades with details")
print("2. Run production sim for Q1 2016 only - save all trades with details")
print("3. Compare trade-by-trade:")
print("   - Which pairs traded")
print("   - Entry prices (should be same day's close)")
print("   - Position sizes (should be same formula)")
print("   - Exit prices and reasons")
print("   - Capital progression")
print()
print("Key questions:")
print("- Are trades opened in the same order?")
print("- Are position sizes exactly the same?")
print("- Are exit prices exactly the same?")
print("- Is capital updated the same way?")
print()
