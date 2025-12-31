"""
Compare predictions: Backtest vs Simulator
===========================================
This will help us understand why results diverge
"""
import pandas as pd
import numpy as np
from production_simulation import ProductionSimulation, calculate_features
from mock_broker_api import MockBrokerAPI
import pickle

print("="*100)
print("PREDICTION COMPARISON: BACKTEST VS SIMULATOR")
print("="*100)
print()

# Focus on a specific problematic period
TEST_DATE = pd.Timestamp('2022-01-03 14:00:00', tz='UTC')  # First trading day of bad quarter (6am PST = 2pm UTC)
PAIR = 'EURUSD'

print(f"Testing date: {TEST_DATE.date()}")
print(f"Pair: {PAIR}")
print()

# ============================================================================
# 1. SIMULATOR PREDICTION (what production would generate)
# ============================================================================
print("1. SIMULATOR PREDICTION:")
print("-" * 60)

api = MockBrokerAPI(data_dir='data')

# Get history up to test date (what simulator sees)
history = api.get_history(PAIR, count=999999, end_date=TEST_DATE)
print(f"History available: {len(history)} days")
print(f"Last date: {history.index[-1].date()}")

# Calculate features
features = calculate_features(history)
features = features.dropna()
print(f"Features after dropna: {len(features)} rows")
print(f"Last feature date: {features.index[-1].date()}")
print()

# Show the actual features for the last day
print("Last day features:")
feature_cols = ['return_1d', 'return_5d', 'ema_10', 'ema_20', 'ema_50', 
                'macd', 'rsi', 'volatility_10d', 'high_20d', 'low_20d']
for col in feature_cols:
    if col in features.columns:
        val = features[col].iloc[-1]
        print(f"  {col:20s}: {val:.6f}")
print()

# ============================================================================
# 2. BACKTEST PREDICTION (from pkl file)
# ============================================================================
print("2. BACKTEST PREDICTION (from quarterly pkl):")
print("-" * 60)

try:
    with open('model_predictions_quarterly.pkl', 'rb') as f:
        all_preds = pickle.load(f)
    
    # Find which quarter contains this date
    quarter = '2022Q1'
    if quarter in all_preds and PAIR in all_preds[quarter]:
        backtest_preds = all_preds[quarter][PAIR]
        
        if TEST_DATE in backtest_preds.index:
            row = backtest_preds.loc[TEST_DATE]
            print(f"Found prediction for {TEST_DATE.date()}:")
            print(f"  High prob: {row['breakout_high_prob']:.4f}")
            print(f"  Low prob:  {row['breakout_low_prob']:.4f}")
            print(f"  High_20d:  {row['high_20d']:.6f}")
            print(f"  Low_20d:   {row['low_20d']:.6f}")
            print(f"  Close:     {row['close']:.6f}")
        else:
            print(f"Date {TEST_DATE.date()} not in backtest predictions")
            print(f"Available dates: {backtest_preds.index[0].date()} to {backtest_preds.index[-1].date()}")
    else:
        print(f"Quarter {quarter} or pair {PAIR} not in predictions file")
        
except FileNotFoundError:
    print("model_predictions_quarterly.pkl not found")
    print("You need to run generate_predictions_quarterly.py first")

print()

# ============================================================================
# 3. COMPARISON OF FEATURE CALCULATION
# ============================================================================
print("3. FEATURE CALCULATION COMPARISON:")
print("-" * 60)

# Load raw data and calculate features the backtest way
file_path = f'data/{PAIR}_1day_with_spreads.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Calculate features on FULL dataset (backtest approach)
df_full = calculate_features(df.copy())

# Get features for same date from both methods
# Remove timezone from TEST_DATE for comparison with tz-naive data
test_date_naive = TEST_DATE.tz_localize(None) if TEST_DATE.tz else TEST_DATE

if test_date_naive in df_full.index:
    print(f"Full dataset features for {test_date_naive.date()}:")
    for col in feature_cols:
        if col in df_full.columns:
            val = df_full.loc[test_date_naive, col]
            print(f"  {col:20s}: {val:.6f}")
    print()
    
    print("Comparison (Simulator vs Backtest):")
    for col in feature_cols:
        if col in features.columns and col in df_full.columns:
            sim_val = features[col].iloc[-1]
            back_val = df_full.loc[test_date_naive, col]
            diff = abs(sim_val - back_val)
            match = "✓" if diff < 0.0001 else "✗"
            print(f"  {col:20s}: {sim_val:12.6f} vs {back_val:12.6f} | diff: {diff:.8f} {match}")
else:
    print(f"Date {test_date_naive.date()} not found in full dataset")

print()
print("="*100)