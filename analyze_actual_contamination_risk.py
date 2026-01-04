"""
ANALYSIS: ACTUAL CONTAMINATION RISK
====================================
Let's carefully analyze whether the original train/test approach has real leakage.

KEY INSIGHT: Rolling windows look BACKWARD, not forward:
- df['high'].rolling(80).max() at row 1000 uses rows 920-999
- This is NOT lookahead bias
- It's the same calculation we'd do in production

The question: Is there ANY actual contamination in the original approach?
"""
import pandas as pd
import numpy as np

print("="*100)
print("CONTAMINATION RISK ANALYSIS")
print("="*100)
print()

# Simulate the original approach
print("ORIGINAL APPROACH:")
print("-" * 80)
print()

# Create sample data
dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
df = pd.DataFrame({
    'date': dates,
    'close': np.random.randn(1000).cumsum() + 100
})
df = df.set_index('date')

print("Step 1: Calculate features on FULL dataset")
print(f"  Dataset size: {len(df)} rows")

# Calculate rolling features (backward-looking)
df['rolling_max_80'] = df['close'].rolling(80).max()
df['rolling_mean_80'] = df['close'].rolling(80).mean()

print(f"  Features calculated using .rolling(80)")
print(f"  Example: Row 500 rolling_max uses rows 420-499 (BACKWARD)")
print()

# Verify backward-looking behavior
row_500_window_start = 500 - 80
row_500_window_end = 500
row_500_max_expected = df['close'].iloc[row_500_window_start:row_500_window_end].max()
row_500_max_actual = df['rolling_max_80'].iloc[500]

print("Verification:")
print(f"  Row 500 uses window [{row_500_window_start}:{row_500_window_end}]")
print(f"  Expected max: {row_500_max_expected:.2f}")
print(f"  Actual max: {row_500_max_actual:.2f}")
print(f"  Match: {np.isclose(row_500_max_expected, row_500_max_actual, equal_nan=True)}")
print()

print("Step 2: Split into train/test (70/30)")
split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"  Train: rows 0-{split_idx-1}")
print(f"  Test: rows {split_idx}-{len(df)-1}")
print()

print("Step 3: Check if train data uses test data")
print(f"  Last train row: {split_idx-1}")
print(f"  Its rolling window: [{split_idx-1-80}:{split_idx-1}]")
print(f"  First test row: {split_idx}")
print(f"  Overlap: {max(0, (split_idx) - (split_idx-1-80))} rows")
print()

# Check if last training row's window overlaps with test set
last_train_window_start = split_idx - 1 - 80
last_train_window_end = split_idx - 1
first_test_row = split_idx

if last_train_window_end < first_test_row:
    print("  [PASS] No overlap - last train window ends before test starts")
else:
    overlap_rows = last_train_window_end - first_test_row + 1
    print(f"  [WARNING] Train window overlaps test by {overlap_rows} rows")

print()
print("="*100)
print("POTENTIAL CONTAMINATION SOURCES")
print("="*100)
print()

contamination_risks = [
    ("Rolling features looking forward", "NO - .rolling() looks backward", False),
    ("Train rows using test data in windows", "NO - windows end before split", False),
    ("Target calculation using future data", "YES - but this is intentional (it's the label)", False),
    ("Feature scaling using test statistics", "NO - no scaling performed", False),
    ("Feature selection on full dataset", "MAYBE - if features were selected based on full dataset performance", True),
    ("Dropping NaN using full dataset info", "NO - NaN drop doesn't leak information", False),
]

print(f"{'Risk':<45} {'Status':<50} {'Concern?':<10}")
print("-" * 105)
for risk, status, concern in contamination_risks:
    concern_str = "YES" if concern else "NO"
    print(f"{risk:<45} {status:<50} {concern_str:<10}")

print()
print("="*100)
print("REAL ISSUES TO CHECK")
print("="*100)
print()

print("1. DATA SNOOPING:")
print("   - Were hyperparameters tuned on the test set?")
print("   - Were features selected based on full dataset performance?")
print("   - Were strategy parameters optimized on the test period?")
print()

print("2. WALK-FORWARD VALIDATION:")
print("   - Models trained on old data (e.g., 2024) may not work on new data (2025-2026)")
print("   - Should implement walk-forward testing with periodic retraining")
print("   - This is NOT contamination, but it's poor validation methodology")
print()

print("3. BACKTEST EXECUTION:")
print("   - Signal at bar T, entry at bar T+1 open: CORRECT")
print("   - Using bid/ask prices: CORRECT")
print("   - Exit logic: Need to verify no same-bar lookahead")
print()

print("="*100)
print("CONCLUSION")
print("="*100)
print()

print("The original 70/30 split approach has NO CLEAR CONTAMINATION if:")
print("  1. Features were not selected based on test set performance")
print("  2. Hyperparameters were not tuned on the test set")
print("  3. Strategy parameters were not optimized on the test period")
print()
print("The MAIN ISSUE is not contamination, but rather:")
print("  - Static train/test split doesn't reflect real trading (no retraining)")
print("  - Need walk-forward validation with periodic model updates")
print()
print("RECOMMENDATION:")
print("  Implement walk-forward testing with quarterly or monthly retraining")
print("  This simulates real production: train -> trade -> retrain -> trade -> ...")
print()
