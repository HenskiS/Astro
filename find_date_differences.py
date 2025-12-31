"""
Find which dates are in one dataset but not the other
"""
import pickle
import pandas as pd

pair = 'NZDUSD'

with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    prod_preds = pickle.load(f)

print("="*100)
print(f"FINDING DATE DIFFERENCES FOR {pair} 2016Q1")
print("="*100)
print()

back_df = backtest_preds['2016Q1'][pair]
prod_df = prod_preds['2016Q1'][pair]

# Normalize indices for comparison
back_dates_normalized = set(d.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
                           for d in back_df.index)
prod_dates_normalized = set(d.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
                           for d in prod_df.index)

print(f"Backtest:   {len(back_dates_normalized)} unique dates")
print(f"Production: {len(prod_dates_normalized)} unique dates")
print()

# Find differences
only_in_back = back_dates_normalized - prod_dates_normalized
only_in_prod = prod_dates_normalized - back_dates_normalized

print(f"Only in BACKTEST: {len(only_in_back)}")
for date in sorted(only_in_back)[:10]:
    print(f"  {date.date()}")
print()

print(f"Only in PRODUCTION: {len(only_in_prod)}")
for date in sorted(only_in_prod)[:10]:
    print(f"  {date.date()}")
print()

# Check if these match the "worst mismatches"
worst_dates = ['2016-03-23', '2016-01-03', '2016-03-14', '2016-03-10', '2016-03-24',
               '2016-03-11']

print("Checking worst mismatch dates:")
for date_str in worst_dates:
    date = pd.Timestamp(date_str).replace(tzinfo=None)
    in_back = date in back_dates_normalized
    in_prod = date in prod_dates_normalized
    print(f"  {date_str}: Backtest={in_back}, Production={in_prod}")
print()
