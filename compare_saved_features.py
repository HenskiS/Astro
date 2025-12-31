"""
Compare the actual saved features from both prediction sets
"""
import pickle
import pandas as pd

# Load predictions
with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    production = pickle.load(f)

# Pick a date where predictions differ significantly
date_str = '2016-01-08'
pair = 'USDCAD'
quarter = '2016Q1'

print("="*100)
print(f"COMPARING SAVED FEATURES FOR {pair} on {date_str}")
print("="*100)
print()

# Get backtest data for this date
backtest_df = backtest[quarter][pair]
# Normalize timezone for comparison
if backtest_df.index.tz is not None:
    backtest_df_naive = backtest_df.copy()
    backtest_df_naive.index = backtest_df_naive.index.tz_localize(None)
else:
    backtest_df_naive = backtest_df

# Find matching date
target_date = pd.Timestamp(date_str)
backtest_rows = backtest_df_naive[backtest_df_naive.index.normalize() == target_date]

if len(backtest_rows) > 0:
    back_row = backtest_rows.iloc[0]
    print("BACKTEST:")
    print(f"  Date: {backtest_rows.index[0]}")
    print(f"  high_20d:  {back_row['high_20d']:.6f}")
    print(f"  low_20d:   {back_row['low_20d']:.6f}")
    print(f"  close:     {back_row['close']:.6f}")
    print(f"  high_prob: {back_row['breakout_high_prob']:.6f}")
    print(f"  low_prob:  {back_row['breakout_low_prob']:.6f}")
else:
    print(f"No backtest data for {date_str}")

print()

# Get production data for this date
production_df = production[quarter][pair]
production_rows = production_df[production_df.index.normalize() == target_date]

if len(production_rows) > 0:
    prod_row = production_rows.iloc[0]
    print("PRODUCTION:")
    print(f"  Date: {production_rows.index[0]}")
    print(f"  high_20d:  {prod_row['high_20d']:.6f}")
    print(f"  low_20d:   {prod_row['low_20d']:.6f}")
    print(f"  close:     {prod_row['close']:.6f}")
    print(f"  high_prob: {prod_row['breakout_high_prob']:.6f}")
    print(f"  low_prob:  {prod_row['breakout_low_prob']:.6f}")
else:
    print(f"No production data for {date_str}")

print()

if len(backtest_rows) > 0 and len(production_rows) > 0:
    print("DIFFERENCES:")
    print(f"  high_20d diff:  {abs(back_row['high_20d'] - prod_row['high_20d']):.6f}")
    print(f"  low_20d diff:   {abs(back_row['low_20d'] - prod_row['low_20d']):.6f}")
    print(f"  close diff:     {abs(back_row['close'] - prod_row['close']):.6f}")
    print(f"  high_prob diff: {abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob']):.6f}")
    print(f"  low_prob diff:  {abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob']):.6f}")
    print()

    if abs(back_row['close'] - prod_row['close']) > 0.0001:
        print("[!] CLOSE prices differ - they're using different date's data!")
    elif abs(back_row['high_20d'] - prod_row['high_20d']) > 0.0001:
        print("[!] Features differ even though CLOSE matches - feature calculation differs!")
    elif abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob']) > 0.01:
        print("[!] Predictions differ even though features match - models are different!")

print()
