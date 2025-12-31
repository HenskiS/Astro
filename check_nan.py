"""
Compare signal generation: How many trades per day?
"""
import pandas as pd
import pickle

print("="*100)
print("SIGNAL GENERATION COMPARISON")
print("="*100)
print()

# Load backtest predictions
with open('model_predictions_quarterly.pkl', 'rb') as f:
    all_preds = pickle.load(f)

# Analyze 2021Q4 (where divergence starts)
q = all_preds['2021Q4']

print("2021Q4 Analysis:")
print("-" * 60)
print(f"Pairs: {len(q)}")

# Count signals per day
all_dates = set()
for pair_df in q.values():
    all_dates.update(pair_df.index)
all_dates = sorted(list(all_dates))

print(f"Trading days: {len(all_dates)}")
print()

# Count how many signals above 0.70 per day
MIN_CONFIDENCE = 0.70
signals_per_day = []

for date in all_dates:
    daily_signals = 0
    for pair, pair_df in q.items():
        if date in pair_df.index:
            row = pair_df.loc[date]
            max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
            if max_prob >= MIN_CONFIDENCE:
                daily_signals += 1
    signals_per_day.append(daily_signals)

print("Signals per day (probability >= 0.70):")
print(f"  Average: {sum(signals_per_day)/len(signals_per_day):.1f}")
print(f"  Min: {min(signals_per_day)}")
print(f"  Max: {max(signals_per_day)}")
print(f"  Total signals: {sum(signals_per_day)}")
print()

print("Total trades from backtest output: 272")
print()

# Check a specific high-signal day
max_signal_day_idx = signals_per_day.index(max(signals_per_day))
max_signal_date = all_dates[max_signal_day_idx]

print(f"Highest signal day: {max_signal_date.date()} with {signals_per_day[max_signal_day_idx]} signals")
print("Pairs with signals on that day:")
for pair, pair_df in q.items():
    if max_signal_date in pair_df.index:
        row = pair_df.loc[max_signal_date]
        max_prob = max(row['breakout_high_prob'], row['breakout_low_prob'])
        if max_prob >= MIN_CONFIDENCE:
            direction = 'HIGH' if row['breakout_high_prob'] > row['breakout_low_prob'] else 'LOW'
            print(f"  {pair}: {max_prob:.3f} ({direction})")

print()
print("="*100)
print()
print("KEY INSIGHT:")
print("If simulator shows 150+ open positions but backtest only has ~3 signals/day,")
print("then simulator is either:")
print("  1. Not closing positions quickly enough (exits differ)")
print("  2. Generating different predictions (model mismatch)")
print("  3. Opening multiple positions per pair (shouldn't happen)")