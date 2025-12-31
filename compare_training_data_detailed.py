"""
Compare training data in detail - check if backtest and production
train on the exact same data for 2016Q1
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI

pair = 'USDCAD'

print("="*100)
print(f"COMPARING TRAINING DATA FOR {pair} - 2016Q1")
print("="*100)
print()

# Training period for 2016Q1: 2010-01-01 to 2015-12-21
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

# BACKTEST APPROACH
print("1. BACKTEST TRAINING DATA:")
print("-"*100)
df = pd.read_csv(f'data/{pair}_1day_with_spreads.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Calculate features on FULL dataset
df_with_features = calculate_features(df.copy())
df_with_features = create_targets(df_with_features)

# Filter to training window
train_df_backtest = df_with_features[(df_with_features.index >= train_start) &
                                      (df_with_features.index <= train_end)].copy()
train_df_backtest = train_df_backtest.dropna()

print(f"  Rows: {len(train_df_backtest)}")
print(f"  Date range: {train_df_backtest.index[0].date()} to {train_df_backtest.index[-1].date()}")
print(f"  target_breakout_high sum: {train_df_backtest['target_breakout_high'].sum()}")
print(f"  target_breakout_low sum: {train_df_backtest['target_breakout_low'].sum()}")
print(f"  First 5 feature values (ema_20): {train_df_backtest['ema_20'].head().tolist()}")
print(f"  Last 5 feature values (ema_20): {train_df_backtest['ema_20'].tail().tolist()}")
print()

# PRODUCTION APPROACH
print("2. PRODUCTION TRAINING DATA:")
print("-"*100)
api = MockBrokerAPI(data_dir='data')
training_data_prod = api.get_history(pair, count=999999, end_date=train_end)

# Calculate features
training_data_prod = calculate_features(training_data_prod)
training_data_prod = create_targets(training_data_prod)

# Filter to training window
training_data_prod = training_data_prod[(training_data_prod.index >= train_start) &
                                         (training_data_prod.index <= train_end)]
training_data_prod = training_data_prod.dropna()

print(f"  Rows: {len(training_data_prod)}")
print(f"  Date range: {training_data_prod.index[0].date()} to {training_data_prod.index[-1].date()}")
print(f"  target_breakout_high sum: {training_data_prod['target_breakout_high'].sum()}")
print(f"  target_breakout_low sum: {training_data_prod['target_breakout_low'].sum()}")
print(f"  First 5 feature values (ema_20): {training_data_prod['ema_20'].head().tolist()}")
print(f"  Last 5 feature values (ema_20): {training_data_prod['ema_20'].tail().tolist()}")
print()

# COMPARISON
print("3. COMPARISON:")
print("-"*100)
if len(train_df_backtest) == len(training_data_prod):
    print(f"  Row count: MATCH ({len(train_df_backtest)} rows)")
else:
    print(f"  Row count: DIFFER (backtest={len(train_df_backtest)}, production={len(training_data_prod)})")

if train_df_backtest['target_breakout_high'].sum() == training_data_prod['target_breakout_high'].sum():
    print(f"  HIGH targets: MATCH ({train_df_backtest['target_breakout_high'].sum()})")
else:
    print(f"  HIGH targets: DIFFER (backtest={train_df_backtest['target_breakout_high'].sum()}, production={training_data_prod['target_breakout_high'].sum()})")

if train_df_backtest['target_breakout_low'].sum() == training_data_prod['target_breakout_low'].sum():
    print(f"  LOW targets: MATCH ({train_df_backtest['target_breakout_low'].sum()})")
else:
    print(f"  LOW targets: DIFFER (backtest={train_df_backtest['target_breakout_low'].sum()}, production={training_data_prod['target_breakout_low'].sum()})")

# Check if features match for common dates
common_dates = train_df_backtest.index.intersection(training_data_prod.index)
if len(common_dates) > 0:
    feature_matches = (train_df_backtest.loc[common_dates, 'ema_20'] == training_data_prod.loc[common_dates, 'ema_20']).sum()
    print(f"  Feature matches: {feature_matches}/{len(common_dates)} dates")
else:
    print(f"  No common dates!")

print()
