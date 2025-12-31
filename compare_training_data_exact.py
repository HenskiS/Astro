"""
Compare the exact training data (features + targets) between backtest and production
for a specific quarter to identify remaining differences
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI
import numpy as np

pair = 'USDCAD'
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print(f"COMPARING EXACT TRAINING DATA FOR {pair} - 2016Q1")
print("="*100)
print()

# BACKTEST APPROACH
print("1. BACKTEST:")
print("-"*100)
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df_with_features = calculate_features(df.copy())
df_with_targets = create_targets(df_with_features)
train_df_backtest = df_with_targets[(df_with_targets.index >= train_start) &
                                     (df_with_targets.index <= train_end)].copy()
train_df_backtest = train_df_backtest.dropna()

print(f"  Rows: {len(train_df_backtest)}")
print(f"  HIGH targets: {train_df_backtest['target_breakout_high'].sum()}")
print(f"  LOW targets: {train_df_backtest['target_breakout_low'].sum()}")
print(f"  Features: {len([c for c in train_df_backtest.columns if c not in ['target_breakout_high', 'target_breakout_low', 'open', 'high', 'low', 'close', 'volume']])}")
print()

# PRODUCTION APPROACH (with fix)
print("2. PRODUCTION (with +10 days fix):")
print("-"*100)
api = MockBrokerAPI(data_dir='data')
data_end_date = train_end + pd.Timedelta(days=10)
training_data = api.get_history(pair, count=999999, end_date=data_end_date)
training_data = calculate_features(training_data)
training_data = create_targets(training_data)
training_data = training_data[(training_data.index >= train_start) &
                              (training_data.index <= train_end)]
training_data = training_data.dropna()

print(f"  Rows: {len(training_data)}")
print(f"  HIGH targets: {training_data['target_breakout_high'].sum()}")
print(f"  LOW targets: {training_data['target_breakout_low'].sum()}")
print(f"  Features: {len([c for c in training_data.columns if c not in ['target_breakout_high', 'target_breakout_low', 'open', 'high', 'low', 'close', 'volume']])}")
print()

# DETAILED COMPARISON
print("3. DETAILED COMPARISON:")
print("-"*100)

# Check if dates match
common_dates = train_df_backtest.index.intersection(training_data.index)
print(f"  Common dates: {len(common_dates)}/{len(train_df_backtest)}")

if len(common_dates) == len(train_df_backtest):
    # Check targets
    high_matches = (train_df_backtest.loc[common_dates, 'target_breakout_high'] ==
                   training_data.loc[common_dates, 'target_breakout_high']).sum()
    low_matches = (train_df_backtest.loc[common_dates, 'target_breakout_low'] ==
                  training_data.loc[common_dates, 'target_breakout_low']).sum()

    print(f"  HIGH targets match: {high_matches}/{len(common_dates)}")
    print(f"  LOW targets match: {low_matches}/{len(common_dates)}")

    # Get feature columns
    feature_cols_back = [c for c in train_df_backtest.columns if c not in
                        ['target_breakout_high', 'target_breakout_low', 'open', 'high', 'low', 'close', 'volume']]
    feature_cols_prod = [c for c in training_data.columns if c not in
                        ['target_breakout_high', 'target_breakout_low', 'open', 'high', 'low', 'close', 'volume']]

    # Check if feature columns match
    if set(feature_cols_back) == set(feature_cols_prod):
        print(f"  Feature columns: MATCH ({len(feature_cols_back)} features)")

        # Check feature column order
        if feature_cols_back == feature_cols_prod:
            print(f"  Feature column ORDER: MATCH")
        else:
            print(f"  Feature column ORDER: DIFFER")
            print(f"    Backtest first 5: {feature_cols_back[:5]}")
            print(f"    Production first 5: {feature_cols_prod[:5]}")

        # Check feature values for common features
        common_features = [f for f in feature_cols_back if f in feature_cols_prod]
        max_diff = 0
        worst_feature = None

        for feat in common_features:
            diff = np.abs(train_df_backtest.loc[common_dates, feat].values -
                         training_data.loc[common_dates, feat].values).max()
            if diff > max_diff:
                max_diff = diff
                worst_feature = feat

        print(f"  Feature values: Max diff = {max_diff:.10f} ({worst_feature})")

        if max_diff < 1e-8:
            print(f"  [SUCCESS] Features are IDENTICAL!")
        else:
            print(f"  [ISSUE] Features differ slightly")
    else:
        print(f"  Feature columns: DIFFER")
        missing_in_prod = set(feature_cols_back) - set(feature_cols_prod)
        missing_in_back = set(feature_cols_prod) - set(feature_cols_back)
        if missing_in_prod:
            print(f"    Missing in production: {missing_in_prod}")
        if missing_in_back:
            print(f"    Missing in backtest: {missing_in_back}")
else:
    print(f"  [ERROR] Dates don't match!")

print()
