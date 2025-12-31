"""
Check what dates are available in saved predictions
"""
import pickle
import pandas as pd

pair = 'NZDUSD'
target_date = pd.Timestamp('2016-03-23')

with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

print("="*100)
print(f"CHECKING DATES FOR {pair}")
print("="*100)
print()

back_df = backtest_preds['2016Q1'][pair]
prod_df = prod_preds['2016Q1'][pair]

print(f"Backtest has {len(back_df)} predictions for 2016Q1 {pair}")
print(f"Production has {len(prod_df)} predictions for 2016Q1 {pair}")
print()

# Check if target date is in either
print(f"Looking for {target_date.date()}:")
back_match = back_df[back_df.index.normalize() == target_date.normalize()]
prod_match = prod_df[prod_df.index.normalize() == target_date.normalize()]

print(f"  Backtest:   {len(back_match)} matches")
if len(back_match) > 0:
    print(f"    Date: {back_match.index[0]}")
    print(f"    HIGH: {back_match.iloc[0]['breakout_high_prob']:.6f}")
    print(f"    LOW:  {back_match.iloc[0]['breakout_low_prob']:.6f}")

print(f"  Production: {len(prod_match)} matches")
if len(prod_match) > 0:
    print(f"    Date: {prod_match.index[0]}")
    print(f"    HIGH: {prod_match.iloc[0]['breakout_high_prob']:.6f}")
    print(f"    LOW:  {prod_match.iloc[0]['breakout_low_prob']:.6f}")
print()

# Show dates around target
print("Dates around 2016-03-23:")
nearby_back = back_df[(back_df.index.normalize() >= target_date.normalize() - pd.Timedelta(days=3)) &
                      (back_df.index.normalize() <= target_date.normalize() + pd.Timedelta(days=3))]
nearby_prod = prod_df[(prod_df.index.normalize() >= target_date.normalize() - pd.Timedelta(days=3)) &
                      (prod_df.index.normalize() <= target_date.normalize() + pd.Timedelta(days=3))]

print("Backtest:")
for date in nearby_back.index:
    print(f"  {date.date()}")

print("\nProduction:")
for date in nearby_prod.index:
    print(f"  {date.date()}")
print()
