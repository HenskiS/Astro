"""
Compare features between backtest and production for a specific date
"""
import pandas as pd
from production_simulation import calculate_features
from mock_broker_api import MockBrokerAPI

# Test date from the comparison output (where predictions differ)
test_date = pd.Timestamp('2016-01-08')
pair = 'USDCAD'

print("="*100)
print(f"COMPARING FEATURES FOR {pair} on {test_date.date()}")
print("="*100)
print()

# BACKTEST APPROACH (full dataset)
print("1. BACKTEST APPROACH (features on full dataset):")
print("-"*100)
df_full = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df_full['date'] = pd.to_datetime(df_full['date'])
df_full = df_full.set_index('date')
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

df_full = calculate_features(df_full)

# Normalize test_date to match df_full timezone
test_date_normalized = test_date if test_date.tz is None else test_date.tz_localize(None)

# Find matching date (handle potential time component differences)
matching_dates = df_full.index[df_full.index.normalize() == test_date_normalized.normalize()]
if len(matching_dates) > 0:
    row_full = df_full.loc[matching_dates[0]]
    print(f"  return_1d: {row_full['return_1d']:.6f}")
    print(f"  ema_20:    {row_full['ema_20']:.6f}")
    print(f"  rsi:       {row_full['rsi']:.6f}")
    print(f"  high_20d:  {row_full['high_20d']:.6f}")
    print(f"  low_20d:   {row_full['low_20d']:.6f}")
else:
    print(f"  Date {test_date_normalized} not in backtest data")
    # Show nearby dates
    nearby = df_full.index[df_full.index >= test_date_normalized][:3]
    if len(nearby) > 0:
        print(f"  Next available dates: {[d.date() for d in nearby]}")
print()

# PRODUCTION APPROACH (only past data)
print("2. PRODUCTION APPROACH (features up to current date):")
print("-"*100)
api = MockBrokerAPI(data_dir='data')
history = api.get_history(pair, count=999999, end_date=test_date)
history_features = calculate_features(history)
history_features = history_features.dropna()

if len(history_features) > 0:
    row_prod = history_features.iloc[-1]
    print(f"  return_1d: {row_prod['return_1d']:.6f}")
    print(f"  ema_20:    {row_prod['ema_20']:.6f}")
    print(f"  rsi:       {row_prod['rsi']:.6f}")
    print(f"  high_20d:  {row_prod['high_20d']:.6f}")
    print(f"  low_20d:   {row_prod['low_20d']:.6f}")
    print(f"  Last date: {history_features.index[-1].date()}")
else:
    print("  No features available")
print()

# COMPARISON
print("3. COMPARISON:")
print("-"*100)
if len(matching_dates) > 0 and len(history_features) > 0:
    features_to_check = ['return_1d', 'ema_20', 'rsi', 'high_20d', 'low_20d']
    all_match = True
    for feat in features_to_check:
        if feat in row_full.index and feat in row_prod.index:
            diff = abs(row_full[feat] - row_prod[feat])
            match = "✓" if diff < 0.0001 else "✗"
            print(f"  {feat:15s}: diff={diff:.8f} {match}")
            if diff >= 0.0001:
                all_match = False

    print()
    if all_match:
        print("[SUCCESS] Features match! Issue must be in model training/prediction.")
    else:
        print("[ISSUE] Features differ! This explains prediction differences.")
else:
    print("Cannot compare - missing data")

print()
