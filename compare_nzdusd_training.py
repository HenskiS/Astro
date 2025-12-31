"""
Compare NZDUSD training data between backtest and production
"""
import pandas as pd
from production_simulation import calculate_features, create_targets
from mock_broker_api import MockBrokerAPI

pair = 'NZDUSD'
train_start = pd.Timestamp('2010-01-01')
train_end = pd.Timestamp('2015-12-21')

print("="*100)
print(f"COMPARING {pair} TRAINING DATA FOR 2016Q1")
print("="*100)
print()

# BACKTEST
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

# PRODUCTION (with fix #1 updated to +15 days)
api = MockBrokerAPI(data_dir='data')
data_end_date = train_end + pd.Timedelta(days=15)
training_data = api.get_history(pair, count=999999, end_date=data_end_date)
training_data = calculate_features(training_data)
training_data = create_targets(training_data)
training_data = training_data[(training_data.index >= train_start) &
                              (training_data.index <= train_end)]
training_data = training_data.dropna()

print(f"Backtest:")
print(f"  Rows: {len(train_df_backtest)}")
print(f"  HIGH targets: {train_df_backtest['target_breakout_high'].sum()}")
print(f"  LOW targets:  {train_df_backtest['target_breakout_low'].sum()}")
print()

print(f"Production:")
print(f"  Rows: {len(training_data)}")
print(f"  HIGH targets: {training_data['target_breakout_high'].sum()}")
print(f"  LOW targets:  {training_data['target_breakout_low'].sum()}")
print()

print("Comparison:")
print("-"*100)
if train_df_backtest['target_breakout_high'].sum() == training_data['target_breakout_high'].sum():
    print("  HIGH targets: MATCH")
else:
    diff = abs(train_df_backtest['target_breakout_high'].sum() - training_data['target_breakout_high'].sum())
    print(f"  HIGH targets: DIFFER by {diff}")

if train_df_backtest['target_breakout_low'].sum() == training_data['target_breakout_low'].sum():
    print("  LOW targets: MATCH")
else:
    diff = abs(train_df_backtest['target_breakout_low'].sum() - training_data['target_breakout_low'].sum())
    print(f"  LOW targets: DIFFER by {diff}")
print()

# If they differ, find which rows
if train_df_backtest['target_breakout_high'].sum() != training_data['target_breakout_high'].sum():
    common_dates = train_df_backtest.index.intersection(training_data.index)
    if len(common_dates) == len(train_df_backtest):
        high_diffs = train_df_backtest.loc[common_dates, 'target_breakout_high'] != training_data.loc[common_dates, 'target_breakout_high']
        num_diffs = high_diffs.sum()
        print(f"  {num_diffs} rows have different HIGH targets")
        if num_diffs > 0 and num_diffs <= 10:
            print("  Dates with differences:")
            for date in common_dates[high_diffs]:
                print(f"    {date.date()}")
print()
