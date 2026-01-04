"""Quick diagnostic of backtest behavior"""
import pandas as pd
import numpy as np
import pickle

# Load predictions
with open('test_predictions_15m_continuous.pkl', 'rb') as f:
    predictions = pickle.load(f)

# Count high-confidence signals across all pairs
total_signals = 0
for pair, preds in predictions.items():
    high_signals = (preds['breakout_high_prob'] > 0.80).sum()
    low_signals = (preds['breakout_low_prob'] > 0.80).sum()
    pair_total = high_signals + low_signals
    total_signals += pair_total
    print(f"{pair}: {pair_total:,} signals ({100*pair_total/len(preds):.2f}%)")

print()
print(f"Total signals across all pairs: {total_signals:,}")
print(f"Avg signals per 15-min bar: {total_signals / len(preds):.2f}")
print()

# Estimate max concurrent if we hold for 24 bars
avg_hold_time = 24  # bars
signal_rate_per_bar = total_signals / len(list(predictions.values())[0])
estimated_concurrent = signal_rate_per_bar * avg_hold_time
print(f"Estimated max concurrent positions (if holding 24 bars): {estimated_concurrent:.1f}")
print()

# Check signal distribution over time
eurusd_preds = predictions['EURUSD']
eurusd_preds['high_signal'] = (eurusd_preds['breakout_high_prob'] > 0.80).astype(int)
eurusd_preds['low_signal'] = (eurusd_preds['breakout_low_prob'] > 0.80).astype(int)
eurusd_preds['any_signal'] = (eurusd_preds['high_signal'] + eurusd_preds['low_signal']).clip(0, 1)

# Count signals by month
eurusd_preds_copy = eurusd_preds.copy()
eurusd_preds_copy['month'] = eurusd_preds_copy.index.to_period('M')
monthly_signals = eurusd_preds_copy.groupby('month')['any_signal'].sum()

print("EURUSD signals by month (first 12 months):")
for month, count in monthly_signals.head(12).items():
    print(f"  {month}: {count} signals")
print()

# Check win probability
print("Average prediction probabilities (all pairs):")
for pair, preds in predictions.items():
    avg_high = preds['breakout_high_prob'].mean()
    avg_low = preds['breakout_low_prob'].mean()
    max_high = preds['breakout_high_prob'].max()
    max_low = preds['breakout_low_prob'].max()
    print(f"  {pair}: avg_high={avg_high:.3f}, avg_low={avg_low:.3f}, max_high={max_high:.3f}, max_low={max_low:.3f}")
