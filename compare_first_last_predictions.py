"""
COMPARE FIRST AND LAST PREDICTIONS
===================================
Check if prediction differences are consistent or changing over time
"""
import pandas as pd
import pickle
import numpy as np

print("="*100)
print("FIRST & LAST 10 PREDICTIONS COMPARISON")
print("="*100)
print()

# Load predictions
with open('model_predictions_quarterly.pkl', 'rb') as f:
    backtest_preds = pickle.load(f)
with open('model_predictions_production.pkl', 'rb') as f:
    production_preds = pickle.load(f)

# Get first quarter (2016Q1)
first_quarter = '2016Q1'
# Get a middle quarter
mid_quarter = '2020Q1'
# Get last quarter
last_quarter = '2025Q4'

def compare_quarter_predictions(quarter_key, n=10):
    """Compare first N predictions in a quarter"""
    print(f"\n{'='*100}")
    print(f"QUARTER: {quarter_key}")
    print(f"{'='*100}\n")

    if quarter_key not in backtest_preds or quarter_key not in production_preds:
        print(f"Quarter {quarter_key} not found in both prediction sets")
        return

    backtest_q = backtest_preds[quarter_key]
    production_q = production_preds[quarter_key]

    # Get common pairs
    common_pairs = set(backtest_q.keys()) & set(production_q.keys())

    if not common_pairs:
        print("No common pairs found")
        return

    # Take first pair for comparison
    pair = sorted(common_pairs)[0]

    print(f"Pair: {pair}")
    print(f"-"*100)

    # Get dataframes
    back_df = backtest_q[pair].copy()
    prod_df = production_q[pair].copy()

    # Remove timezone from backtest
    if back_df.index.tz is not None:
        back_df.index = back_df.index.tz_localize(None)

    # Get common dates
    common_dates = sorted(set(back_df.index.normalize()) & set(prod_df.index.normalize()))

    if len(common_dates) < n:
        print(f"Only {len(common_dates)} common dates found")
        n = len(common_dates)

    print(f"\nFIRST {n} PREDICTIONS:")
    print("-"*100)
    for i, date in enumerate(common_dates[:n]):
        back_row = back_df.loc[back_df.index.normalize() == date].iloc[0]
        prod_row = prod_df.loc[prod_df.index.normalize() == date].iloc[0]

        high_diff = abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob'])
        low_diff = abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob'])

        print(f"{i+1}. {date.date()}")
        print(f"   High: Back={back_row['breakout_high_prob']:.4f}, Prod={prod_row['breakout_high_prob']:.4f}, Diff={high_diff:.4f}")
        print(f"   Low:  Back={back_row['breakout_low_prob']:.4f}, Prod={prod_row['breakout_low_prob']:.4f}, Diff={low_diff:.4f}")

    print(f"\nLAST {n} PREDICTIONS:")
    print("-"*100)
    for i, date in enumerate(common_dates[-n:]):
        back_row = back_df.loc[back_df.index.normalize() == date].iloc[0]
        prod_row = prod_df.loc[prod_df.index.normalize() == date].iloc[0]

        high_diff = abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob'])
        low_diff = abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob'])

        print(f"{i+1}. {date.date()}")
        print(f"   High: Back={back_row['breakout_high_prob']:.4f}, Prod={prod_row['breakout_high_prob']:.4f}, Diff={high_diff:.4f}")
        print(f"   Low:  Back={back_row['breakout_low_prob']:.4f}, Prod={prod_row['breakout_low_prob']:.4f}, Diff={low_diff:.4f}")

    # Calculate average difference for this quarter
    all_diffs = []
    for date in common_dates:
        back_row = back_df.loc[back_df.index.normalize() == date].iloc[0]
        prod_row = prod_df.loc[prod_df.index.normalize() == date].iloc[0]
        high_diff = abs(back_row['breakout_high_prob'] - prod_row['breakout_high_prob'])
        low_diff = abs(back_row['breakout_low_prob'] - prod_row['breakout_low_prob'])
        all_diffs.append(max(high_diff, low_diff))

    print(f"\nQUARTER STATS:")
    print(f"  Mean difference: {np.mean(all_diffs):.6f}")
    print(f"  Max difference: {np.max(all_diffs):.6f}")
    print(f"  Within 1% tolerance: {sum(1 for d in all_diffs if d < 0.01)}/{len(all_diffs)} ({sum(1 for d in all_diffs if d < 0.01)/len(all_diffs)*100:.1f}%)")

# Compare each quarter
compare_quarter_predictions(first_quarter)
compare_quarter_predictions(mid_quarter)
compare_quarter_predictions(last_quarter)

print()
