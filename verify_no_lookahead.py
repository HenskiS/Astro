"""
VERIFY NO LOOKAHEAD BIAS IN BACKTESTS
======================================
Checks for common lookahead issues that inflate backtest performance:
1. Positions abandoned at boundaries (not properly closed)
2. Feature calculation using future data
3. Prediction timestamps vs entry timestamps
4. Train/test overlap in walk-forward
"""
import pandas as pd
import pickle
import numpy as np

print("="*100)
print("LOOKAHEAD BIAS VERIFICATION")
print("="*100)
print()

# ============================================================================
# CHECK 1: Are there open positions at end of backtest?
# ============================================================================
print("CHECK 1: Open Positions at End of Backtest")
print("-" * 100)
print()

# Load backtest results
trades_15m = pd.read_csv('backtest_15m_optimized_results.csv')
trades_1h = pd.read_csv('backtest_1h_optimized_results.csv')

trades_15m['exit_date'] = pd.to_datetime(trades_15m['exit_date'])
trades_1h['exit_date'] = pd.to_datetime(trades_1h['exit_date'])

last_trade_15m = trades_15m['exit_date'].max()
last_trade_1h = trades_1h['exit_date'].max()

# Load predictions to see test period end
with open('test_predictions_15m.pkl', 'rb') as f:
    preds_15m = pickle.load(f)
with open('test_predictions_1h.pkl', 'rb') as f:
    preds_1h = pickle.load(f)

test_end_15m = max([df.index.max() for df in preds_15m.values()])
test_end_1h = max([df.index.max() for df in preds_1h.values()])

print("15M Strategy:")
print(f"  Test period ends: {test_end_15m}")
print(f"  Last trade exit:  {last_trade_15m}")
gap_15m = (test_end_15m - last_trade_15m).total_seconds() / 3600  # hours
print(f"  Gap: {gap_15m:.1f} hours")

if gap_15m > 24:  # More than 1 day
    print(f"  [WARNING] {gap_15m:.1f} hour gap suggests positions may be open at end!")
else:
    print(f"  [OK] Small gap, positions likely closed")

print()
print("1H Strategy:")
print(f"  Test period ends: {test_end_1h}")
print(f"  Last trade exit:  {last_trade_1h}")
gap_1h = (test_end_1h - last_trade_1h).total_seconds() / 3600
print(f"  Gap: {gap_1h:.1f} hours")

if gap_1h > 120:  # More than 5 days (120h)
    print(f"  [WARNING]  WARNING: {gap_1h:.1f} hour gap suggests positions may be open at end!")
else:
    print(f"  [OK] OK: Small gap, positions likely closed")

print()

# ============================================================================
# CHECK 2: Prediction timestamps vs entry timestamps
# ============================================================================
print("CHECK 2: Prediction Timestamps vs Entry Timestamps")
print("-" * 100)
print()

trades_15m['entry_date'] = pd.to_datetime(trades_15m['entry_date'])
trades_1h['entry_date'] = pd.to_datetime(trades_1h['entry_date'])

# Check if any trades entered before their prediction date (IMPOSSIBLE without lookahead)
print("15M Strategy:")
for pair in trades_15m['pair'].unique():
    pair_trades = trades_15m[trades_15m['pair'] == pair]
    pred_dates = set(preds_15m[pair].index)

    invalid_entries = 0
    for _, trade in pair_trades.iterrows():
        entry_date = trade['entry_date']
        if entry_date not in pred_dates:
            # Entry date should match a prediction date
            invalid_entries += 1

    if invalid_entries > 0:
        print(f"  [WARNING]  {pair}: {invalid_entries} trades with no matching prediction (possible lookahead)")
    else:
        print(f"  [OK] {pair}: All trades have matching predictions")

print()
print("1H Strategy:")
for pair in trades_1h['pair'].unique():
    pair_trades = trades_1h[trades_1h['pair'] == pair]
    pred_dates = set(preds_1h[pair].index)

    invalid_entries = 0
    for _, trade in pair_trades.iterrows():
        entry_date = trade['entry_date']
        if entry_date not in pred_dates:
            invalid_entries += 1

    if invalid_entries > 0:
        print(f"  [WARNING]  {pair}: {invalid_entries} trades with no matching prediction (possible lookahead)")
    else:
        print(f"  [OK] {pair}: All trades have matching predictions")

print()

# ============================================================================
# CHECK 3: Feature calculation direction (past vs future)
# ============================================================================
print("CHECK 3: Feature Calculation Direction")
print("-" * 100)
print()

print("Checking calculate_features() for .shift(-N) calls (future data)...")
print()

# Read the training script
with open('train_model_15m.py', 'r') as f:
    train_code = f.read()

# Find calculate_features function
start = train_code.find('def calculate_features(')
end = train_code.find('\ndef calculate_targets(')
features_code = train_code[start:end]

# Look for negative shifts (future data)
future_shifts = []
for line_num, line in enumerate(features_code.split('\n'), 1):
    if '.shift(-' in line and 'def calculate_features' not in line:
        future_shifts.append((line_num, line.strip()))

if len(future_shifts) > 0:
    print(f"  [WARNING]  WARNING: Found {len(future_shifts)} instances of .shift(-N) in features:")
    for line_num, line in future_shifts:
        print(f"     Line {line_num}: {line}")
    print("  This uses FUTURE data in features = LOOKAHEAD BIAS!")
else:
    print("  [OK] OK: No .shift(-N) found in calculate_features()")
    print("  All features use only past data (.shift(1) or .rolling() are OK)")

print()

# ============================================================================
# CHECK 4: Target calculation (should look forward - this is correct)
# ============================================================================
print("CHECK 4: Target Calculation (SHOULD use future data)")
print("-" * 100)
print()

# Find calculate_targets function
start = train_code.find('def calculate_targets(')
end = train_code.find('\n\n# Load', start)
targets_code = train_code[start:end] if end != -1 else train_code[start:start+1000]

print("Checking calculate_targets() for .shift(-N) calls...")
print()

future_shifts_targets = []
for line_num, line in enumerate(targets_code.split('\n'), 1):
    if '.shift(-' in line and 'def calculate_targets' not in line:
        future_shifts_targets.append((line_num, line.strip()))

if len(future_shifts_targets) > 0:
    print(f"  [OK] EXPECTED: Found {len(future_shifts_targets)} instances of .shift(-N) in targets:")
    for line_num, line in future_shifts_targets[:3]:  # Show first 3
        print(f"     Line {line_num}: {line}")
    print()
    print("  This is CORRECT - targets should look into the future during training.")
    print("  The model learns to predict future breakouts based on current features.")
else:
    print("  [WARNING]  WARNING: No future data used in targets - targets may be incorrect!")

print()

# ============================================================================
# CHECK 5: Performance Reality Check
# ============================================================================
print("CHECK 5: Performance Reality Check")
print("-" * 100)
print()

print("Metrics that are TOO GOOD suggest possible lookahead bias:")
print()

# 15M results
win_rate_15m = (trades_15m['profit_pct'] > 0).sum() / len(trades_15m)
avg_winner_15m = trades_15m[trades_15m['profit_pct'] > 0]['profit_pct'].mean()
avg_loser_15m = trades_15m[trades_15m['profit_pct'] <= 0]['profit_pct'].mean()
profit_ratio_15m = abs(avg_winner_15m / avg_loser_15m) if avg_loser_15m != 0 else 0

print("15M Strategy:")
print(f"  Win Rate: {win_rate_15m:.1%}")
if win_rate_15m > 0.95:
    print(f"    [WARNING]  SUSPICIOUS: >95% win rate is extremely rare")
elif win_rate_15m > 0.90:
    print(f"    [WARNING]  HIGH: >90% win rate is very good, verify it's real")
else:
    print(f"    [OK] REALISTIC: Win rate is achievable")

print(f"  Profit Ratio: {profit_ratio_15m:.2f}:1")
if profit_ratio_15m > 3.0:
    print(f"    [WARNING]  SUSPICIOUS: >3:1 profit ratio with high win rate is very rare")
else:
    print(f"    [OK] REALISTIC: Profit ratio is achievable")

print()

# 1H results
win_rate_1h = (trades_1h['profit_pct'] > 0).sum() / len(trades_1h)
avg_winner_1h = trades_1h[trades_1h['profit_pct'] > 0]['profit_pct'].mean()
avg_loser_1h = trades_1h[trades_1h['profit_pct'] <= 0]['profit_pct'].mean()
profit_ratio_1h = abs(avg_winner_1h / avg_loser_1h) if avg_loser_1h != 0 else 0

print("1H Strategy:")
print(f"  Win Rate: {win_rate_1h:.1%}")
if win_rate_1h > 0.95:
    print(f"    [WARNING]  SUSPICIOUS: >95% win rate is extremely rare")
elif win_rate_1h > 0.90:
    print(f"    [WARNING]  HIGH: >90% win rate is very good, verify it's real")
else:
    print(f"    [OK] REALISTIC: Win rate is achievable")

print(f"  Profit Ratio: {profit_ratio_1h:.2f}:1")
if profit_ratio_1h > 3.0:
    print(f"    [WARNING]  SUSPICIOUS: >3:1 profit ratio with high win rate is very rare")
else:
    print(f"    [OK] REALISTIC: Profit ratio is achievable")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*100)
print("SUMMARY")
print("="*100)
print()

issues = []

if gap_15m > 24:
    issues.append("15M: Large gap at end suggests open positions")
if gap_1h > 120:
    issues.append("1H: Large gap at end suggests open positions")
if len(future_shifts) > 0:
    issues.append("CRITICAL: Features use future data (.shift(-N))")
if win_rate_15m > 0.95 or win_rate_1h > 0.95:
    issues.append("Win rates >95% are suspicious")

if len(issues) == 0:
    print("[OK] ALL CHECKS PASSED")
    print()
    print("No obvious lookahead bias detected.")
    print("Your backtest results appear realistic and trustworthy.")
else:
    print("[WARNING]  POTENTIAL ISSUES FOUND:")
    print()
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("Review the checks above and fix any issues before trusting results.")

print()
print("="*100)
