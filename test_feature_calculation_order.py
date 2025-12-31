"""
Test if calculating features on full dataset vs partial dataset gives different results
"""
import pandas as pd
from production_simulation import calculate_features

pair = 'USDCAD'
test_date = pd.Timestamp('2016-01-08')

print("="*100)
print(f"TESTING FEATURE CALCULATION ORDER FOR {pair} on {test_date.date()}")
print("="*100)
print()

# Load full dataset
df_full = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df_full['date'] = pd.to_datetime(df_full['date'])
df_full = df_full.set_index('date')
if df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)

# APPROACH 1: Calculate features on FULL dataset, then filter (BACKTEST)
print("1. BACKTEST APPROACH (features on full dataset, then filter):")
print("-"*100)
df_full_features = calculate_features(df_full.copy())
test_date_normalized = test_date if test_date.tz is None else test_date.tz_localize(None)
matching_dates = df_full_features.index[df_full_features.index.normalize() == test_date_normalized.normalize()]

if len(matching_dates) > 0:
    row_full = df_full_features.loc[matching_dates[0]]
    print(f"  ema_20:        {row_full['ema_20']:.10f}")
    print(f"  rsi:           {row_full['rsi']:.10f}")
    print(f"  high_20d:      {row_full['high_20d']:.10f}")
    print(f"  low_20d:       {row_full['low_20d']:.10f}")
    print(f"  volatility_10d: {row_full['volatility_10d']:.10f}")
else:
    print("  Date not found")
print()

# APPROACH 2: Filter first, then calculate features (PRODUCTION)
print("2. PRODUCTION APPROACH (filter to past data, then calculate features):")
print("-"*100)
df_partial = df_full[df_full.index <= test_date].copy()
df_partial_features = calculate_features(df_partial)

if len(df_partial_features) > 0:
    row_partial = df_partial_features.iloc[-1]
    print(f"  ema_20:        {row_partial['ema_20']:.10f}")
    print(f"  rsi:           {row_partial['rsi']:.10f}")
    print(f"  high_20d:      {row_partial['high_20d']:.10f}")
    print(f"  low_20d:       {row_partial['low_20d']:.10f}")
    print(f"  volatility_10d: {row_partial['volatility_10d']:.10f}")
    print(f"  Last date:     {df_partial_features.index[-1].date()}")
else:
    print("  No features available")
print()

# COMPARISON
print("3. COMPARISON:")
print("-"*100)
if len(matching_dates) > 0 and len(df_partial_features) > 0:
    features_to_check = ['ema_20', 'rsi', 'high_20d', 'low_20d', 'volatility_10d', 'macd', 'atr']
    all_match = True

    for feat in features_to_check:
        if feat in row_full.index and feat in row_partial.index:
            diff = abs(row_full[feat] - row_partial[feat])
            match = "✓" if diff < 1e-8 else "✗"
            print(f"  {feat:15s}: diff={diff:.15f} {match}")
            if diff >= 1e-8:
                all_match = False

    print()
    if all_match:
        print("[SUCCESS] Feature calculation order doesn't matter!")
    else:
        print("[ISSUE] Feature calculation order DOES matter - this explains prediction differences!")
else:
    print("Cannot compare - missing data")

print()
