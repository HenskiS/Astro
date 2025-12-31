"""
Compare the exact features used for making a prediction between backtest and production
"""
import pandas as pd
import pickle
from production_simulation import calculate_features
from mock_broker_api import MockBrokerAPI

# One of the worst mismatches
pair = 'NZDUSD'
prediction_date = pd.Timestamp('2016-03-23')

print("="*100)
print(f"COMPARING PREDICTION FEATURES FOR {pair} ON {prediction_date.date()}")
print("="*100)
print()

# Load saved predictions to see what was used
with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

back_df = backtest_preds['2016Q1'][pair]
prod_df = prod_preds['2016Q1'][pair]

# Get the saved predictions
back_df_naive = back_df.copy()
if back_df_naive.index.tz is not None:
    back_df_naive.index = back_df_naive.index.tz_localize(None)

back_row = back_df_naive.loc[back_df_naive.index.normalize() == prediction_date.normalize()].iloc[0]
prod_row = prod_df.loc[prod_df.index.normalize() == prediction_date.normalize()].iloc[0]

print("SAVED PREDICTIONS:")
print("-"*100)
print(f"Backtest:")
print(f"  Date: {back_row.name}")
print(f"  HIGH prob: {back_row['breakout_high_prob']:.6f}")
print(f"  LOW prob:  {back_row['breakout_low_prob']:.6f}")
print(f"  high_20d:  {back_row['high_20d']:.6f}")
print(f"  low_20d:   {back_row['low_20d']:.6f}")
print(f"  close:     {back_row['close']:.6f}")
print()

print(f"Production:")
print(f"  Date: {prod_row.name}")
print(f"  HIGH prob: {prod_row['breakout_high_prob']:.6f}")
print(f"  LOW prob:  {prod_row['breakout_low_prob']:.6f}")
print(f"  high_20d:  {prod_row['high_20d']:.6f}")
print(f"  low_20d:   {prod_row['low_20d']:.6f}")
print(f"  close:     {prod_row['close']:.6f}")
print()

print("DIFFERENCES:")
print("-"*100)
print(f"  HIGH prob: {abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob']):.6f}")
print(f"  LOW prob:  {abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob']):.6f}")
print(f"  high_20d:  {abs(back_row['high_20d'] - prod_row['high_20d']):.6f}")
print(f"  low_20d:   {abs(back_row['low_20d'] - prod_row['low_20d']):.6f}")
print(f"  close:     {abs(back_row['close'] - prod_row['close']):.6f}")
print()

# Key observation
if abs(back_row['high_20d'] - prod_row['high_20d']) > 0.0001 or abs(back_row['low_20d'] - prod_row['low_20d']) > 0.0001:
    print("[ISSUE] Features used for prediction DIFFER - models saw different inputs!")
    print("This means either:")
    print("  1. Different date's data was used (fix #2 not applied)")
    print("  2. Feature calculation differs between backtest and production")
elif abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob']) > 0.01:
    print("[ISSUE] Features match but predictions differ - models are different!")
    print("This means:")
    print("  1. Training data differed (fix #1 not fully applied)")
    print("  2. Or there's non-determinism in XGBoost")
else:
    print("[OK] Small difference - likely acceptable")
print()
