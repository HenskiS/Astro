"""
COMPARE SAVED PREDICTIONS
=========================
Compare predictions from production simulation vs quarterly backtest.

This will show us exactly where and how the predictions differ.
"""
import pandas as pd
import pickle
import numpy as np

print("="*100)
print("COMPARING SAVED PREDICTIONS")
print("="*100)
print()

# Load both prediction files
print("Loading predictions...")
try:
    with open('model_predictions_quarterly.pkl', 'rb') as f:
        backtest_preds = pickle.load(f)
    print(f"  [OK] Loaded quarterly backtest predictions")
except FileNotFoundError:
    print("  [ERROR] model_predictions_quarterly.pkl not found - run generate_predictions_quarterly.py first")
    exit(1)

try:
    with open('model_predictions_production.pkl', 'rb') as f:
        production_preds = pickle.load(f)
    print(f"  [OK] Loaded production simulation predictions")
except FileNotFoundError:
    print("  [ERROR] model_predictions_production.pkl not found - run production_simulation.py first")
    exit(1)

print()

# Get common quarters
backtest_quarters = set(backtest_preds.keys())
production_quarters = set(production_preds.keys())
common_quarters = sorted(backtest_quarters & production_quarters)

print(f"Backtest quarters: {len(backtest_quarters)}")
print(f"Production quarters: {len(production_quarters)}")
print(f"Common quarters: {len(common_quarters)}")
print()

if len(common_quarters) == 0:
    print("No common quarters to compare!")
    exit(1)

# Compare predictions for each quarter
all_comparisons = []

for quarter in common_quarters:
    backtest_quarter = backtest_preds[quarter]
    production_quarter = production_preds[quarter]

    # Get common pairs
    backtest_pairs = set(backtest_quarter.keys())
    production_pairs = set(production_quarter.keys())
    common_pairs = backtest_pairs & production_pairs

    for pair in common_pairs:
        backtest_df = backtest_quarter[pair]
        production_df = production_quarter[pair]

        # Remove timezone from backtest dates for comparison (UTC -> naive)
        backtest_df_naive = backtest_df.copy()
        if backtest_df_naive.index.tz is not None:
            backtest_df_naive.index = backtest_df_naive.index.tz_localize(None)

        # Get common dates (both timezone-naive now)
        backtest_dates = set(d.normalize() for d in backtest_df_naive.index)
        production_dates = set(d.normalize() for d in production_df.index)
        common_dates = backtest_dates & production_dates

        for date in common_dates:
            # Get predictions for this date
            try:
                backtest_row = backtest_df_naive.loc[backtest_df_naive.index.normalize() == date].iloc[0]
                production_row = production_df.loc[production_df.index.normalize() == date].iloc[0]
            except:
                continue

            # Compare probabilities
            high_diff = abs(backtest_row['breakout_high_prob'] - production_row['breakout_high_prob'])
            low_diff = abs(backtest_row['breakout_low_prob'] - production_row['breakout_low_prob'])

            all_comparisons.append({
                'quarter': quarter,
                'pair': pair,
                'date': date,
                'backtest_high': backtest_row['breakout_high_prob'],
                'production_high': production_row['breakout_high_prob'],
                'backtest_low': backtest_row['breakout_low_prob'],
                'production_low': production_row['breakout_low_prob'],
                'high_diff': high_diff,
                'low_diff': low_diff,
                'max_diff': max(high_diff, low_diff)
            })

# Convert to DataFrame
comparison_df = pd.DataFrame(all_comparisons)

print("="*100)
print("COMPARISON RESULTS")
print("="*100)
print()

print(f"Total comparisons: {len(comparison_df)}")
print()

if len(comparison_df) == 0:
    print("[ERROR] NO COMPARISONS FOUND!")
    print()
    print("Possible reasons:")
    print("  1. Production simulation hasn't generated predictions yet")
    print("  2. Date/timezone mismatch between backtest and production predictions")
    print("  3. No common quarters/pairs/dates between the two prediction sets")
    print()

    # Show what we have
    print("Backtest quarters:", sorted(backtest_quarters))
    print("Production quarters:", sorted(production_quarters))
    exit(1)

# Summary statistics
print("High Probability Differences:")
print(f"  Mean:   {comparison_df['high_diff'].mean():.6f}")
print(f"  Median: {comparison_df['high_diff'].median():.6f}")
print(f"  Max:    {comparison_df['high_diff'].max():.6f}")
print(f"  Min:    {comparison_df['high_diff'].min():.6f}")
print()

print("Low Probability Differences:")
print(f"  Mean:   {comparison_df['low_diff'].mean():.6f}")
print(f"  Median: {comparison_df['low_diff'].median():.6f}")
print(f"  Max:    {comparison_df['low_diff'].max():.6f}")
print(f"  Min:    {comparison_df['low_diff'].min():.6f}")
print()

# Check how many are within tolerance
tolerance = 0.01  # 1%
within_tolerance = (comparison_df['max_diff'] < tolerance).sum()
print(f"Predictions within {tolerance:.2%} tolerance: {within_tolerance}/{len(comparison_df)} ({within_tolerance/len(comparison_df)*100:.1f}%)")
print()

# Show worst mismatches
print("Worst 10 Mismatches:")
print("-"*100)
worst = comparison_df.nlargest(10, 'max_diff')
for idx, row in worst.iterrows():
    print(f"{row['quarter']} | {row['pair']} | {row['date'].date()}")
    print(f"  High: Backtest={row['backtest_high']:.4f}, Production={row['production_high']:.4f}, Diff={row['high_diff']:.4f}")
    print(f"  Low:  Backtest={row['backtest_low']:.4f}, Production={row['production_low']:.4f}, Diff={row['low_diff']:.4f}")
    print()

# Check if predictions are essentially identical
if comparison_df['max_diff'].mean() < 0.001:
    print("="*100)
    print("[SUCCESS] PREDICTIONS MATCH - Implementations are equivalent!")
    print("="*100)
else:
    print("="*100)
    print("[INFO] PREDICTIONS DIFFER - Investigating causes...")
    print("="*100)
    print()

    # Analyze patterns
    print("Analyzing patterns:")

    # By quarter
    print("\nMean difference by quarter:")
    by_quarter = comparison_df.groupby('quarter')['max_diff'].mean().sort_values(ascending=False)
    for quarter, diff in by_quarter.head(5).items():
        print(f"  {quarter}: {diff:.6f}")

    # By pair
    print("\nMean difference by pair:")
    by_pair = comparison_df.groupby('pair')['max_diff'].mean().sort_values(ascending=False)
    for pair, diff in by_pair.head(5).items():
        print(f"  {pair}: {diff:.6f}")

print()
